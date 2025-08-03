"""
하수도 막힘 실시간 감지 FastAPI 백엔드
서버와 프론트엔드 분리 버전
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from collections import deque
from datetime import datetime, timedelta
import asyncio
import json
import logging
import math
import cv2
import base64
import numpy as np
import threading

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="하수도 막힘 감지 시스템 API", version="2.0.0")

# 전역 상태 보호를 위한 Lock
status_lock = threading.RLock()  # 재진입 가능한 Lock
detections_lock = threading.RLock()
clients_lock = threading.RLock()

# CORS 설정 (프론트엔드 분리로 인해 필요)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DetectionData(BaseModel):
    timestamp: str
    garbage_type: str
    confidence: float
    bbox: List[int]  # [x1, y1, x2, y2]
    area: float
    location: str = "main_pipe"

class AlertData(BaseModel):
    level: str  # "safe", "warning", "danger"
    message: str
    timestamp: datetime
    risk_score: float

class StatusResponse(BaseModel):
    risk_level: str
    risk_score: float
    total_detections: int
    last_detection: Optional[str]
    alerts_today: int
    pipe_status: str
    blockage_percentage: Optional[float] = 0.0
    garbage_volume: Optional[float] = 0.0
    flow_restriction: Optional[str] = "없음"
    accumulated_areas: Optional[int] = 0

class BlockageAnalysis(BaseModel):
    blockage_percentage: float
    garbage_volume: float
    flow_restriction: str
    accumulated_areas: int
    total_area: float

class AIAnalysis(BaseModel):
    """AI 분석 결과"""
    risk_assessment: str  # "low", "medium", "high", "critical"
    confidence_level: float  # AI 분석 신뢰도 (0-1)
    reasoning: str  # AI 분석 근거
    recommendations: List[str]  # AI 권장사항
    false_positive_probability: float  # 오탐지 확률
    trend_analysis: str  # 추세 분석
    severity_score: float  # AI 심각도 점수 (0-100)

current_status = {
    "risk_score": 0.0,
    "risk_level": "safe",
    "total_detections": 0,
    "last_detection": None,
    "alerts_today": 0,
    "pipe_status": "정상 - 원활한 흐름",
    "accumulation_rate": 0.0,
    "blockage_percentage": 0.0,
    "garbage_volume": 0.0,
    "flow_restriction": "없음",
    "accumulated_areas": 0
}

recent_detections: deque = deque(maxlen=100)
recent_alerts: deque = deque(maxlen=50)
connected_clients: List[WebSocket] = []

# 2초 간격 처리를 위한 새로운 상태 관리
pending_detections: Dict[str, Dict] = {}  # 임시 저장용
confirmed_detections: deque = deque(maxlen=100)  # 2초 확정된 감지들
CONFIRMATION_TIME_SECONDS = 2  # 2초 확정 시간

# 비디오 스트리밍
current_frame = None
camera_active = False

# 새로운 위험도 임계값 (다층적 평가 기준)
RISK_THRESHOLDS = {
    "safe": (0, 30),
    "warning": (31, 60),
    "caution": (61, 75),
    "danger": (76, 100)
}

# 쓰레기 유형별 위험도 가중치 (막힘 위험성 기준)
GARBAGE_RISK_WEIGHTS = {
    # 높은 막힘 위험 (유연한 재질, 큰 부피)
    'plastic_bag': 1.8,
    'garbage_bag': 1.8,
    'plastic_film': 1.7,
    'cloth': 1.6,
    'tissue': 1.5,
    'paper_bag': 1.4,
    'food_waste': 1.4,
    
    # 중간 막힘 위험
    'paper': 1.2,
    'cardboard': 1.1,
    'plastic_container': 1.0,
    'foam': 1.0,
    
    # 낮은 막힘 위험 (단단한 재질)
    'plastic_bottle': 0.8,
    'glass_bottle': 0.7,
    'metal_can': 0.6,
    'glass': 0.5,
    
    # 기타
    'other': 1.0
}

# AI 분석 가중치 (신뢰도 기반)
AI_WEIGHTS = {
    "low": 0.3,
    "medium": 0.6,
    "high": 0.9,
    "critical": 1.2
}

# 감지 신뢰도 임계값 (낮은 신뢰도도 허용)
MIN_CONFIDENCE_THRESHOLD = 0.06 # 6% 이상 신뢰도만 처리 (매우 관대)
MIN_AREA_THRESHOLD = 500        # 최소 면적 더욱 감소 (더 작은 객체도 감지)
MIN_DETECTIONS_FOR_WARNING = 2  # 경고 발생을 위한 최소 감지 횟수 감소
MIN_DETECTIONS_FOR_DANGER = 4   # 위험 발생을 위한 최소 감지 횟수 감소

# ==================== 분석 함수들 ====================

def get_dynamic_thresholds(weather_risk: float = 1.0, seasonal_factor: float = 1.0, location_factor: float = 1.0) -> Dict[str, tuple]:
    """환경 조건에 따른 동적 임계값 조정"""
    base_thresholds = {
        "safe": (0, 25),
        "warning": (26, 50), 
        "caution": (51, 75),
        "danger": (76, 100)
    }
    
    # 환경 요인을 종합한 조정 계수
    adjustment_factor = weather_risk * seasonal_factor * location_factor
    
    # 임계값 조정 (위험 상황일수록 더 낮은 임계값 적용)
    adjusted_thresholds = {}
    for level, (low, high) in base_thresholds.items():
        if adjustment_factor > 1.2:  # 높은 위험 환경
            adjusted_low = max(0, int(low * 0.8))  # 20% 낮춤
            adjusted_high = max(adjusted_low + 1, int(high * 0.8))
        elif adjustment_factor > 1.0:  # 보통 위험 환경
            adjusted_low = max(0, int(low * 0.9))  # 10% 낮춤
            adjusted_high = max(adjusted_low + 1, int(high * 0.9))
        else:  # 낮은 위험 환경
            adjusted_low = low
            adjusted_high = high
            
        adjusted_thresholds[level] = (adjusted_low, adjusted_high)
    
    return adjusted_thresholds

def get_garbage_type_risk_weight(garbage_type: str) -> float:
    """쓰레기 유형에 따른 위험도 가중치 반환"""
    # 쓰레기 유형 정규화 (다양한 형태의 이름 매핑)
    normalized_type = garbage_type.lower().replace(' ', '_')
    
    # 유형별 매핑
    type_mappings = {
        'plastic': ['plastic_bag', 'plastic_film', 'plastic_container', 'plastic_bottle'],
        'paper': ['paper', 'paper_bag', 'cardboard', 'tissue'],
        'food': ['food_waste', 'organic'],
        'metal': ['metal_can', 'aluminium'],
        'glass': ['glass', 'glass_bottle'],
        'other': ['cloth', 'garbage_bag', 'foam']
    }
    
    # 직접 매칭 시도
    if normalized_type in GARBAGE_RISK_WEIGHTS:
        return GARBAGE_RISK_WEIGHTS[normalized_type]
    
    # 카테고리 기반 매칭
    for category, types in type_mappings.items():
        if any(t in normalized_type for t in types):
            # 해당 카테고리의 평균 가중치 계산
            category_weights = [GARBAGE_RISK_WEIGHTS.get(t, 1.0) for t in types if t in GARBAGE_RISK_WEIGHTS]
            if category_weights:
                return sum(category_weights) / len(category_weights)
    
    # 기본값
    return GARBAGE_RISK_WEIGHTS.get('other', 1.0)

def analyze_spatiotemporal_patterns(detections: List[DetectionData]) -> Dict[str, float]:
    """시공간적 패턴 분석"""
    if not detections:
        return {
            'accumulation_rate': 0.0,
            'concentration_factor': 0.0,
            'persistence_score': 0.0,
            'spatial_clustering': 0.0,
            'temporal_intensity': 0.0
        }
    
    now = datetime.now()
    
    # 1. 축적 속도 계산 (최근 1시간)
    recent_hour_detections = []
    for d in detections:
        try:
            det_time = datetime.fromisoformat(d.timestamp.replace('Z', '+00:00'))
            hours_ago = (now - det_time).total_seconds() / 3600
            if hours_ago <= 1.0:
                recent_hour_detections.append(d)
        except:
            continue
    
    accumulation_rate = len(recent_hour_detections) / max(1, len(detections)) * 100
    
    # 2. 위치별 집중도 분석
    location_clusters = {}
    for d in recent_hour_detections:
        bbox = d.bbox
        # 100x100 픽셀 단위로 그리드 생성
        grid_x = bbox[0] // 100
        grid_y = bbox[1] // 100
        grid_key = f"{grid_x}_{grid_y}"
        
        if grid_key not in location_clusters:
            location_clusters[grid_key] = {
                'count': 0,
                'total_area': 0,
                'types': set()
            }
        
        location_clusters[grid_key]['count'] += 1
        location_clusters[grid_key]['total_area'] += d.area
        location_clusters[grid_key]['types'].add(d.garbage_type)
    
    # 집중도 점수 계산 (클러스터당 평균 감지 수)
    if location_clusters:
        total_detections = sum(cluster['count'] for cluster in location_clusters.values())
        concentration_factor = total_detections / len(location_clusters)
    else:
        concentration_factor = 0.0
    
    # 3. 시간별 지속성 분석 (같은 위치에서의 연속 감지)
    persistence_scores = []
    for grid_key, cluster in location_clusters.items():
        if cluster['count'] >= 3:  # 3회 이상 감지된 위치
            # 해당 위치의 감지 시간 분포 계산
            grid_detections = []
            for d in recent_hour_detections:
                bbox = d.bbox
                if f"{bbox[0]//100}_{bbox[1]//100}" == grid_key:
                    try:
                        det_time = datetime.fromisoformat(d.timestamp.replace('Z', '+00:00'))
                        grid_detections.append(det_time)
                    except:
                        continue
            
            if len(grid_detections) >= 2:
                grid_detections.sort()
                time_span = (grid_detections[-1] - grid_detections[0]).total_seconds() / 3600
                persistence_score = min(time_span * cluster['count'], 10.0)  # 최대 10점
                persistence_scores.append(persistence_score)
    
    avg_persistence = sum(persistence_scores) / len(persistence_scores) if persistence_scores else 0.0
    
    # 4. 공간 클러스터링 점수 (인접한 그리드의 감지 밀도)
    spatial_clustering = 0.0
    for grid_key in location_clusters:
        x, y = map(int, grid_key.split('_'))
        # 주변 8개 셀 확인
        neighbor_count = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                neighbor_key = f"{x+dx}_{y+dy}"
                if neighbor_key in location_clusters:
                    neighbor_count += 1
        
        # 주변 클러스터가 많을수록 높은 점수
        spatial_clustering += neighbor_count * location_clusters[grid_key]['count']
    
    spatial_clustering = min(spatial_clustering / len(location_clusters) if location_clusters else 0, 20.0)
    
    # 5. 시간적 집중도 (단위 시간당 감지 빈도 변화)
    time_intervals = []
    if len(recent_hour_detections) >= 2:
        times = []
        for d in recent_hour_detections:
            try:
                det_time = datetime.fromisoformat(d.timestamp.replace('Z', '+00:00'))
                times.append(det_time)
            except:
                continue
        
        times.sort()
        for i in range(1, len(times)):
            interval = (times[i] - times[i-1]).total_seconds() / 60  # 분 단위
            time_intervals.append(interval)
        
        if time_intervals:
            avg_interval = sum(time_intervals) / len(time_intervals)
            # 간격이 짧을수록 높은 집중도
            temporal_intensity = max(0, 10 - avg_interval) if avg_interval < 10 else 0
        else:
            temporal_intensity = 0.0
    else:
        temporal_intensity = 0.0
    
    return {
        'accumulation_rate': round(accumulation_rate, 2),
        'concentration_factor': round(concentration_factor, 2),
        'persistence_score': round(avg_persistence, 2),
        'spatial_clustering': round(spatial_clustering, 2),
        'temporal_intensity': round(temporal_intensity, 2)
    }

def calculate_environmental_risk_factors() -> Dict[str, float]:
    """환경적 위험 요인 계산 (실제 구현시 외부 API 연동)"""
    # 실제 구현시에는 기상청 API, 계절 정보 등을 활용
    current_month = datetime.now().month
    
    # 계절별 위험 요인
    seasonal_risk = 1.0
    if current_month in [6, 7, 8]:  # 여름 (장마철)
        seasonal_risk = 1.3
    elif current_month in [9, 10, 11]:  # 가을 (낙엽철)
        seasonal_risk = 1.2
    elif current_month in [12, 1, 2]:  # 겨울 (동결)
        seasonal_risk = 1.1
    
    # 시간대별 위험 요인 (출퇴근 시간대 쓰레기 증가)
    current_hour = datetime.now().hour
    time_risk = 1.0
    if current_hour in [7, 8, 9, 17, 18, 19]:  # 출퇴근 시간
        time_risk = 1.1
    
    return {
        'weather_risk': 1.0,  # 실제로는 기상 API에서 가져옴
        'seasonal_factor': seasonal_risk,
        'time_factor': time_risk,
        'location_factor': 1.0  # 실제로는 지역별 특성 반영
    }

def generate_detection_key(detection: DetectionData) -> str:
    """감지 식별키 생성 (위치와 유형 기반)"""
    bbox = detection.bbox
    center_x = (bbox[0] + bbox[2]) // 2
    center_y = (bbox[1] + bbox[3]) // 2
    # 100픽셀 그리드로 그룹화하여 작은 움직임 무시
    grid_x = center_x // 100
    grid_y = center_y // 100
    return f"{detection.garbage_type}_{grid_x}_{grid_y}"

def check_and_confirm_detections():
    """대기 중인 감지들을 확인하고 2초 지속된 것들을 확정"""
    global pending_detections

    with detections_lock:
        now = datetime.now()

        confirmed_keys = []
        for key, detection_info in pending_detections.items():
            first_detected = detection_info['first_detected']
            time_diff = (now - first_detected).total_seconds()

            if time_diff >= CONFIRMATION_TIME_SECONDS:
                # 2초 지속된 감지를 확정
                confirmed_detection = detection_info['detection']
                confirmed_detections.append(confirmed_detection)
                confirmed_keys.append(key)

                logger.info(f"✅ 2초 지속 확정: {confirmed_detection.garbage_type} at {key}")
                return confirmed_detection

        # 확정된 감지들을 대기 목록에서 제거
        for key in confirmed_keys:
            del pending_detections[key]

        return None

def add_to_pending_detections(detection: DetectionData):
    """새로운 감지를 대기 목록에 추가"""
    global pending_detections

    with detections_lock:
        key = generate_detection_key(detection)
        now = datetime.now()

        if key not in pending_detections:
            # 새로운 감지
            pending_detections[key] = {
                'detection': detection,
                'first_detected': now,
                'last_updated': now,
                'count': 1
            }
            logger.debug(f"⏳ 새 감지 대기: {detection.garbage_type} at {key}")
        else:
            # 기존 감지 업데이트
            pending_detections[key]['last_updated'] = now
            pending_detections[key]['count'] += 1
            pending_detections[key]['detection'] = detection  # 최신 데이터로 업데이트
            logger.debug(f"🔄 감지 업데이트: {detection.garbage_type} at {key} (count: {pending_detections[key]['count']})")

def cleanup_old_pending_detections():
    """오래된 대기 감지들을 정리 (5초 이상 업데이트 없음)"""
    global pending_detections

    with detections_lock:
        now = datetime.now()

        old_keys = []
        for key, detection_info in pending_detections.items():
            time_since_update = (now - detection_info['last_updated']).total_seconds()
            if time_since_update > 5.0:  # 5초 이상 업데이트 없음
                old_keys.append(key)

        for key in old_keys:
            logger.debug(f"🗑️ 오래된 대기 감지 제거: {key}")
            del pending_detections[key]

def is_duplicate_detection(new_detection: DetectionData, threshold_seconds: int = 10) -> bool:
    """중복 감지인지 확인"""
    if not recent_detections:
        return False
    
    try:
        new_time = datetime.fromisoformat(new_detection.timestamp.replace('Z', '+00:00'))
        new_bbox = new_detection.bbox
        new_center = ((new_bbox[0] + new_bbox[2]) // 2, (new_bbox[1] + new_bbox[3]) // 2)
        
        for detection in list(recent_detections)[-5:]:
            try:
                det_time = datetime.fromisoformat(detection.timestamp.replace('Z', '+00:00'))
                time_diff = (new_time - det_time).total_seconds()
                
                if time_diff < threshold_seconds:
                    det_bbox = detection.bbox
                    det_center = ((det_bbox[0] + det_bbox[2]) // 2, (det_bbox[1] + det_bbox[3]) // 2)
                    distance = math.sqrt((new_center[0] - det_center[0])**2 + (new_center[1] - det_center[1])**2)
                    
                    if distance < 50 and detection.garbage_type == new_detection.garbage_type:
                        return True
            except:
                continue
                
    except:
        pass
    
    return False

def analyze_pipe_blockage(detections: List[DetectionData]) -> BlockageAnalysis:
    """하수구 막힘 정도 분석"""
    if not detections:
        return BlockageAnalysis(
            blockage_percentage=0.0,
            garbage_volume=0.0,
            flow_restriction="없음",
            accumulated_areas=0,
            total_area=0
        )
    
    now = datetime.now()
    recent_hour = now - timedelta(hours=1)
    
    # 최근 1시간 내 감지 필터링
    recent_detections_list = []
    for d in detections:
        try:
            detection_time = datetime.fromisoformat(d.timestamp.replace('Z', '+00:00'))
            if detection_time >= recent_hour:
                recent_detections_list.append(d)
        except:
            recent_detections_list.append(d)
    
    # 축적 영역 분석
    blockage_areas = {}
    total_area = 0
    
    for detection in recent_detections_list:
        bbox = detection.bbox
        area_key = f"{bbox[0]//100}_{bbox[1]//100}"
        
        if area_key not in blockage_areas:
            blockage_areas[area_key] = {
                "count": 0,
                "total_area": 0,
                "max_confidence": 0,
                "garbage_types": set()
            }
        
        blockage_areas[area_key]["count"] += 1
        blockage_areas[area_key]["total_area"] += detection.area
        blockage_areas[area_key]["max_confidence"] = max(
            blockage_areas[area_key]["max_confidence"], 
            detection.confidence
        )
        blockage_areas[area_key]["garbage_types"].add(detection.garbage_type)
        total_area += detection.area
    
    # 막힘 정도 계산
    pipe_width = 640
    pipe_height = 480
    total_pipe_area = pipe_width * pipe_height
    
    blockage_percentage = min((total_area / total_pipe_area) * 100, 100)
    
    # 흐름 제한 수준 결정
    if blockage_percentage < 10:
        flow_restriction = "미미함"
    elif blockage_percentage < 30:
        flow_restriction = "경미함"
    elif blockage_percentage < 60:
        flow_restriction = "보통"
    elif blockage_percentage < 80:
        flow_restriction = "심각함"
    else:
        flow_restriction = "매우 심각함"
    
    # 용적 추정
    estimated_depth = 5  # cm
    garbage_volume = (total_area / 10000) * estimated_depth  # cm³
    
    return BlockageAnalysis(
        blockage_percentage=round(blockage_percentage, 1),
        garbage_volume=round(garbage_volume, 2),
        flow_restriction=flow_restriction,
        accumulated_areas=len(blockage_areas),
        total_area=total_area
    )

def is_valid_detection(detection: DetectionData) -> bool:
    """감지가 유효한지 검사"""
    # 신뢰도 체크
    if detection.confidence < MIN_CONFIDENCE_THRESHOLD:
        return False
    
    # 최소 면적 체크
    if detection.area < MIN_AREA_THRESHOLD:
        return False
    
    # 바운딩 박스 유효성 체크
    bbox = detection.bbox
    if len(bbox) != 4 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return False
    
    return True

def calculate_risk_score_with_ai(detections: List[DetectionData]) -> tuple[float, AIAnalysis]:
    """단순하고 효과적인 위험도 계산"""
    with status_lock:
        current_risk = current_status.get("risk_score", 0)
    
    # 최근 15초 이내의 감지만 사용 (실시간 반영 강화)
    now = datetime.now()
    recent_detections_only = []
    for d in detections:
        try:
            det_time = datetime.fromisoformat(d.timestamp.replace('Z', '+00:00'))
            seconds_ago = (now - det_time).total_seconds()
            if seconds_ago <= 8:  # 8초 이내만 활성 상태로 간주 (더 빠른 반응)
                recent_detections_only.append(d)
        except:
            continue
    
    # 감지가 없으면 위험도 감소
    if not recent_detections_only:
        decay_rate = 15.0  # 15% 감소 (더 빠른 감소)
        new_score = max(0.0, current_risk - decay_rate)
        
        ai_analysis = AIAnalysis(
            risk_assessment="low",
            confidence_level=0.9,
            reasoning="최근 8초 내 감지된 쓰레기가 없어 안전한 상태입니다.",
            recommendations=["정기적인 모니터링을 계속하세요."],
            false_positive_probability=0.0,
            trend_analysis="개선",
            severity_score=0.0
        )
        return new_score, ai_analysis
    
    # 유효한 감지만 필터링
    valid_detections = [d for d in recent_detections_only if is_valid_detection(d)]
    
    if not valid_detections:
        base_score = max(0, current_risk - 10)  # 더 큰 감소
        ai_analysis = AIAnalysis(
            risk_assessment="low",
            confidence_level=0.8,
            reasoning="유효한 감지가 없어 안전한 상태입니다.",
            recommendations=["감지 시스템을 점검하세요."],
            false_positive_probability=0.3,
            trend_analysis="개선",
            severity_score=0.0
        )
        return base_score, ai_analysis
    
    # === 단순하고 직관적인 위험도 계산 ===
    
    # 1. 기본 점수: 감지 개수에 따른 점수 (가장 중요한 요소)
    detection_count = len(valid_detections)
    base_score = min(detection_count * 8, 60)  # 개수당 8점, 최대 60점
    
    # 2. 신뢰도 보너스 (높은 신뢰도일수록 더 위험)
    avg_confidence = sum(d.confidence for d in valid_detections) / len(valid_detections)
    confidence_bonus = (avg_confidence - 0.3) * 30  # 30% 이상부터 보너스
    confidence_bonus = max(0, min(confidence_bonus, 20))  # 0-20점
    
    # 3. 면적 보너스 (큰 쓰레기일수록 더 위험)
    total_area = sum(d.area for d in valid_detections)
    area_bonus = min(total_area / 5000, 15)  # 면적당 점수, 최대 15점
    
    # 4. 쓰레기 종류 가중치
    type_bonus = 0
    for detection in valid_detections:
        type_weight = get_garbage_type_risk_weight(detection.garbage_type)
        type_bonus += (type_weight - 1.0) * 3  # 기본 1.0에서 벗어난 만큼 점수 추가
    type_bonus = min(type_bonus, 10)  # 최대 10점
    
    # 최종 점수 계산
    calculated_score = base_score + confidence_bonus + area_bonus + type_bonus

    # 점진적 변화 적용 (급격한 변화 방지)
    if calculated_score > current_risk:
        # 증가 시: 차이의 70%만 반영
        change = (calculated_score - current_risk) * 0.7
        new_score = min(current_risk + change, 100.0)
    else:
        # 감소 시: 차이의 50%만 반영 (천천히 감소)
        change = (current_risk - calculated_score) * 0.5
        new_score = max(calculated_score, current_risk - change)

    # 막힘 분석을 위해 status_lock 내에서 current_status 업데이트
    
    # AI 분석 생성 (단순화)
    if new_score >= 75:
        risk_assessment = "critical"
        reasoning = f"다수의 쓰레기가 감지되어 매우 위험합니다. (감지수: {detection_count}, 평균신뢰도: {avg_confidence:.2f})"
    elif new_score >= 50:
        risk_assessment = "high"
        reasoning = f"여러 쓰레기가 감지되어 주의가 필요합니다. (감지수: {detection_count}, 평균신뢰도: {avg_confidence:.2f})"
    elif new_score >= 25:
        risk_assessment = "medium"
        reasoning = f"일부 쓰레기가 감지되었습니다. (감지수: {detection_count}, 평균신뢰도: {avg_confidence:.2f})"
    else:
        risk_assessment = "low"
        reasoning = f"소량의 쓰레기가 감지되었습니다. (감지수: {detection_count}, 평균신뢰도: {avg_confidence:.2f})"
    
    # 권장사항
    recommendations = []
    if new_score >= 60:
        recommendations.append("즉시 정비팀에 연락하세요.")
        recommendations.append("해당 구간의 흐름을 확인하세요.")
    elif new_score >= 30:
        recommendations.append("정기적인 청소를 고려하세요.")
    else:
        recommendations.append("계속 모니터링하세요.")
    
    # 추세 분석
    risk_change = new_score - current_risk
    if risk_change > 3:
        trend_analysis = "악화"
    elif risk_change < -3:
        trend_analysis = "개선"
    else:
        trend_analysis = "안정"
    
    ai_analysis = AIAnalysis(
        risk_assessment=risk_assessment,
        confidence_level=min(avg_confidence, 0.95),
        reasoning=reasoning,
        recommendations=recommendations,
        false_positive_probability=max(0.1, 1.0 - avg_confidence),
        trend_analysis=trend_analysis,
        severity_score=min(new_score, 100.0)
    )
    
    # 상태 업데이트 (필요한 것만)
    blockage_analysis = analyze_pipe_blockage(valid_detections)
    current_status.update({
        "blockage_percentage": blockage_analysis.blockage_percentage,
        "garbage_volume": blockage_analysis.garbage_volume,
        "flow_restriction": blockage_analysis.flow_restriction,
        "accumulated_areas": blockage_analysis.accumulated_areas
    })
    
    return new_score, ai_analysis

def calculate_enhanced_risk_change(detections: List[DetectionData], current_risk: float, patterns: Dict[str, float]) -> float:
    """개선된 동적 위험도 변화 계산"""
    if not detections:
        time_since_last = get_time_since_last_detection()
        decay_rate = 2.0  # 분당 2% 감소
        return -min(decay_rate * time_since_last, current_risk)
    
    # 축적 속도 기반 변화
    accumulation_change = patterns['accumulation_rate'] * 0.3
    
    # 공간적 집중도 기반 변화
    concentration_change = patterns['concentration_factor'] * 0.2
    
    # 시간적 집중도 기반 변화
    temporal_change = patterns['temporal_intensity'] * 0.3
    
    # 지속성 기반 변화
    persistence_change = patterns['persistence_score'] * 0.2
    
    total_change = accumulation_change + concentration_change + temporal_change + persistence_change
    
    # 최대 변화량 제한
    return min(max(total_change, -10), 15)

def calculate_risk_score(detections: List[DetectionData]) -> float:
    """기존 호환성을 위한 래퍼 함수"""
    score, _ = calculate_risk_score_with_ai(detections)
    return score

def get_risk_level(score: float) -> str:
    """새로운 4단계 위험도 레벨 결정"""
    if score >= RISK_THRESHOLDS["danger"][0]:
        return "danger"
    elif score >= RISK_THRESHOLDS["caution"][0]:
        return "caution"
    elif score >= RISK_THRESHOLDS["warning"][0]:
        return "warning"
    else:
        return "safe"

def get_pipe_status(risk_level: str) -> str:
    """새로운 4단계 파이프 상태 텍스트"""
    status_map = {
        "safe": "정상 - 원활한 흐름",
        "warning": "주의 - 축적량 증가", 
        "caution": "경고 - 막힘 위험 증가",
        "danger": "위험 - 막힘 가능성 높음"
    }
    return status_map.get(risk_level, "알 수 없음")

def get_time_since_last_detection() -> float:
    """마지막 감지로부터 경과 시간 (분) - 정확한 타임존 처리"""
    with detections_lock:
        if not recent_detections:
            return 60.0  # 기본값: 60분 (감지가 전혀 없음)

        try:
            last_detection = recent_detections[-1]
            last_time = datetime.fromisoformat(last_detection.timestamp.replace('Z', '+00:00'))

            # 시간대 정보가 없으면 현재 시간대로 가정
            if last_time.tzinfo is None:
                last_time = last_time.replace(tzinfo=datetime.now().astimezone().tzinfo)

            now = datetime.now().astimezone()
            time_diff = (now - last_time).total_seconds() / 60  # 분 단위

            return max(0.0, time_diff)
        except Exception as e:
            logger.warning(f"시간 계산 오류: {e}")
            return 10.0  # 오류시 기본값

def calculate_dynamic_risk_change(detections: List[DetectionData], current_risk: float) -> float:
    """동적 위험도 변화 계산"""
    if not detections:
        # 쓰레기가 없으면 자연 감소
        time_since_last = get_time_since_last_detection()
        decay_rate = 1.5  # 분당 1.5% 감소
        decay_amount = min(decay_rate * time_since_last, current_risk)
        return -decay_amount
    
    # 최근 감지 분석 (최근 5분)
    now = datetime.now()
    recent_detections_5min = []
    
    for d in detections[-10:]:  # 최근 10개만 확인
        try:
            det_time = datetime.fromisoformat(d.timestamp.replace('Z', '+00:00'))
            minutes_ago = (now - det_time).total_seconds() / 60
            if minutes_ago <= 5:  # 5분 이내
                recent_detections_5min.append(d)
        except:
            continue
    
    if not recent_detections_5min:
        # 최근 5분 내 감지가 없으면 감소
        time_since_last = get_time_since_last_detection()
        decay_rate = 1.0  # 분당 1% 감소
        decay_amount = min(decay_rate * time_since_last, current_risk)
        return -decay_amount
    
    # 최근 감지가 있으면 위험도 증가 (더 보수적으로)
    recent_count = len(recent_detections_5min)
    avg_confidence = sum(d.confidence for d in recent_detections_5min) / len(recent_detections_5min)
    
    # 신뢰도가 높고 감지가 많을수록 위험도 증가 (더 제한적으로)
    increase_rate = min(recent_count * 2 * avg_confidence, 8)  # 최대 8% 증가
    
    return increase_rate

def analyze_with_ai(detections: List[DetectionData], blockage_analysis: BlockageAnalysis) -> AIAnalysis:
    """AI 기반 위험도 분석"""
    if not detections:
        return AIAnalysis(
            risk_assessment="low",
            confidence_level=0.9,
            reasoning="감지된 쓰레기가 없어 안전한 상태입니다.",
            recommendations=["정기적인 모니터링을 계속하세요."],
            false_positive_probability=0.0,
            trend_analysis="안정적",
            severity_score=0.0
        )
    
    # 최근 감지 분석 (최근 10개)
    recent_detections = detections[-10:] if len(detections) > 10 else detections
    now = datetime.now()
    
    # 시간 분포 분석
    time_distribution = []
    for d in recent_detections:
        try:
            det_time = datetime.fromisoformat(d.timestamp.replace('Z', '+00:00'))
            minutes_ago = (now - det_time).total_seconds() / 60
            time_distribution.append(minutes_ago)
        except:
            continue
    
    # 신뢰도 분석
    avg_confidence = sum(d.confidence for d in recent_detections) / len(recent_detections)
    high_confidence_count = sum(1 for d in recent_detections if d.confidence > 0.8)
    confidence_ratio = high_confidence_count / len(recent_detections)
    
    # 쓰레기 유형 분석
    garbage_types = {}
    for d in recent_detections:
        garbage_types[d.garbage_type] = garbage_types.get(d.garbage_type, 0) + 1
    
    # AI 분석 로직
    risk_factors = []
    severity_score = 0.0

    # 동적 변화 분석
    with status_lock:
        current_risk = current_status.get("risk_score", 0)
        previous_risk = current_status.get("previous_risk_score", current_risk)
    risk_change = current_risk - previous_risk
    
    # 1. 막힘률 분석 (더 엄격하게)
    if blockage_analysis.blockage_percentage > 80:
        risk_factors.append("막힘률이 80%를 초과하여 매우 심각한 상황")
        severity_score += 40
    elif blockage_analysis.blockage_percentage > 60:
        risk_factors.append("막힘률이 60%를 초과하여 심각한 상황")
        severity_score += 25
    elif blockage_analysis.blockage_percentage > 40:
        risk_factors.append("막힘률이 40%를 초과하여 주의가 필요")
        severity_score += 10
    elif blockage_analysis.blockage_percentage > 20:
        risk_factors.append("막힘률이 20%를 초과하여 모니터링 필요")
        severity_score += 5
    
    # 2. 신뢰도 분석 (더 엄격하게)
    if avg_confidence < 0.8:
        risk_factors.append("평균 신뢰도가 낮아 오탐지 가능성 높음")
        severity_score -= 15  # 신뢰도가 낮으면 위험도 대폭 감소
    elif avg_confidence > 0.95:
        risk_factors.append("매우 높은 신뢰도로 정확한 감지")
        severity_score += 8
    elif avg_confidence > 0.85:
        risk_factors.append("높은 신뢰도로 정확한 감지")
        severity_score += 3
    
    # 3. 시간 분포 분석 (더 엄격하게)
    if time_distribution:
        recent_count = sum(1 for t in time_distribution if t <= 5)  # 5분 이내
        if recent_count >= 5:
            risk_factors.append("최근 5분 내 다수 감지로 급속한 축적")
            severity_score += 25
        elif recent_count >= 3:
            risk_factors.append("최근 5분 내 여러 감지로 축적 증가")
            severity_score += 15
        elif recent_count >= 1:
            risk_factors.append("최근 감지로 지속적 모니터링 필요")
            severity_score += 5
    
    # 4. 쓰레기 유형 분석 (더 엄격하게)
    dangerous_types = ["plastic_bag", "cloth", "paper", "organic"]
    dangerous_count = sum(garbage_types.get(t, 0) for t in dangerous_types)
    if dangerous_count >= 5:
        risk_factors.append("막힘 위험이 높은 쓰레기 다수 감지")
        severity_score += 20
    elif dangerous_count >= 3:
        risk_factors.append("막힘 위험이 높은 쓰레기 여러 개 감지")
        severity_score += 10
    
    # 5. 축적 영역 분석 (더 엄격하게)
    if blockage_analysis.accumulated_areas >= 8:
        risk_factors.append("다수의 축적 영역으로 분산된 막힘")
        severity_score += 15
    elif blockage_analysis.accumulated_areas >= 5:
        risk_factors.append("여러 축적 영역으로 막힘 위험")
        severity_score += 8
    
    # 6. 동적 변화 분석 (더 엄격하게)
    if risk_change > 10:
        risk_factors.append("위험도가 급속히 증가하는 상황")
        severity_score += 20
    elif risk_change > 5:
        risk_factors.append("위험도가 점진적으로 증가")
        severity_score += 10
    elif risk_change < -10:
        risk_factors.append("위험도가 급속히 감소하는 개선 상황")
        severity_score -= 15
    elif risk_change < -5:
        risk_factors.append("위험도가 점진적으로 감소")
        severity_score -= 8
    
    # 7. 추세 분석 (더 엄격하게)
    if len(detections) >= 30:  # 더 많은 데이터 필요
        recent_15 = detections[-15:]
        older_15 = detections[-30:-15]
        recent_avg = sum(d.area for d in recent_15) / len(recent_15)
        older_avg = sum(d.area for d in older_15) / len(older_15)
        
        if recent_avg > older_avg * 2.0:  # 더 큰 증가 필요
            risk_factors.append("쓰레기 크기가 급속히 증가하는 추세")
            severity_score += 15
        elif recent_avg > older_avg * 1.5:
            risk_factors.append("쓰레기 크기가 증가하는 추세")
            severity_score += 8
        elif recent_avg < older_avg * 0.3:  # 더 큰 감소 필요
            risk_factors.append("쓰레기 크기가 급속히 감소하는 추세")
            severity_score -= 10
        elif recent_avg < older_avg * 0.5:
            risk_factors.append("쓰레기 크기가 감소하는 추세")
            severity_score -= 5
    
    # 오탐지 확률 계산 (더 엄격하게)
    false_positive_prob = 0.0
    if avg_confidence < 0.8:
        false_positive_prob = 0.4
    if len(recent_detections) < 5:
        false_positive_prob += 0.3
    if len(recent_detections) < 3:
        false_positive_prob += 0.2
    
    # 위험도 평가 (더 엄격하게)
    if severity_score >= 70:  # 임계값 상향 조정
        risk_assessment = "critical"
        if risk_change > 10:
            reasoning = "다중 위험 요소가 복합적으로 작용하여 매우 위험한 상황이며, 위험도가 급속히 증가하고 있습니다."
        else:
            reasoning = "다중 위험 요소가 복합적으로 작용하여 매우 위험한 상황"
    elif severity_score >= 50:  # 임계값 상향 조정
        risk_assessment = "high"
        if risk_change > 5:
            reasoning = "여러 위험 요소가 확인되어 높은 위험도이며, 위험도가 증가하는 추세입니다."
        else:
            reasoning = "여러 위험 요소가 확인되어 높은 위험도"
    elif severity_score >= 25:  # 임계값 상향 조정
        risk_assessment = "medium"
        if risk_change < -5:
            reasoning = "일부 위험 요소가 확인되어 주의가 필요하지만, 위험도가 감소하는 개선 추세입니다."
        else:
            reasoning = "일부 위험 요소가 확인되어 주의가 필요"
    else:
        risk_assessment = "low"
        if risk_change < -10:
            reasoning = "대부분의 지표가 안전 범위 내에 있으며, 위험도가 급속히 감소하는 좋은 상황입니다."
        elif risk_change < -5:
            reasoning = "대부분의 지표가 안전 범위 내에 있으며, 위험도가 감소하는 개선 추세입니다."
        else:
            reasoning = "대부분의 지표가 안전 범위 내에 있음"
    
    # 권장사항 생성
    recommendations = []
    if risk_assessment in ["critical", "high"]:
        recommendations.append("즉시 정비팀에 연락하여 점검을 요청하세요.")
        recommendations.append("해당 구간의 물 흐름을 모니터링하세요.")
    if blockage_analysis.blockage_percentage > 30:
        recommendations.append("정기적인 청소 일정을 앞당기세요.")
    if avg_confidence < 0.7:
        recommendations.append("카메라 렌즈를 점검하고 정확도를 높이세요.")
    if len(recent_detections) < 5:
        recommendations.append("더 많은 데이터를 수집하여 분석 정확도를 높이세요.")
    
    # 동적 추세 분석
    with status_lock:
        current_risk = current_status.get("risk_score", 0)
        previous_risk = current_status.get("previous_risk_score", current_risk)
    risk_change = current_risk - previous_risk

    if risk_change > 5:
        trend_analysis = "급속 악화"
    elif risk_change > 2:
        trend_analysis = "악화"
    elif risk_change < -5:
        trend_analysis = "급속 개선"
    elif risk_change < -2:
        trend_analysis = "개선"
    elif abs(risk_change) <= 2:
        trend_analysis = "안정"
    else:
        trend_analysis = "불안정"

    # 이전 위험도 저장
    with status_lock:
        current_status["previous_risk_score"] = current_risk
    
    return AIAnalysis(
        risk_assessment=risk_assessment,
        confidence_level=min(avg_confidence, 0.95),
        reasoning=reasoning,
        recommendations=recommendations,
        false_positive_probability=false_positive_prob,
        trend_analysis=trend_analysis,
        severity_score=min(severity_score, 100.0)
    )

def update_status(detections_list: List[DetectionData]):
    """전역 상태 업데이트 (AI 분석 포함)"""
    risk_score, ai_analysis = calculate_risk_score_with_ai(detections_list)

    with status_lock:
        current_status["risk_score"] = risk_score
        current_status["risk_level"] = get_risk_level(risk_score)
        current_status["pipe_status"] = get_pipe_status(current_status["risk_level"])
        current_status["total_detections"] = len(detections_list)
        current_status["ai_analysis"] = ai_analysis.model_dump()

        if detections_list:
            current_status["last_detection"] = detections_list[-1].timestamp
            current_status["accumulation_rate"] = len(detections_list)

async def broadcast_to_clients(data: Dict[str, Any]):
    """WebSocket 브로드캐스트"""
    with clients_lock:
        if not connected_clients:
            return

        clients_to_send = list(connected_clients)

    message = json.dumps(data, default=str)
    disconnected = []

    for client in clients_to_send:
        try:
            await client.send_text(message)
        except Exception as e:
            logger.warning(f"클라이언트 전송 실패: {e}")
            disconnected.append(client)

    with clients_lock:
        for client in disconnected:
            if client in connected_clients:
                connected_clients.remove(client)

# ==================== API 엔드포인트 ====================

@app.post("/detect", summary="쓰레기 감지 데이터 처리")
async def process_detection(data: DetectionData):
    """쓰레기 감지 데이터 처리 - 즉시 처리 방식"""
    try:
        # 1단계: 유효성 검사
        if not is_valid_detection(data):
            logger.debug(f"유효하지 않은 감지 무시: {data.garbage_type} (신뢰도: {data.confidence:.2f}, 면적: {data.area})")
            return {
                "success": True,
                "invalid": True,
                "reason": f"Low confidence ({data.confidence:.2f}) or small area ({data.area})",
                "risk_score": current_status["risk_score"],
                "risk_level": current_status["risk_level"]
            }

        # 2단계: 중복 감지 확인
        if is_duplicate_detection(data):
            logger.debug(f"중복 감지 무시: {data.garbage_type}")
            return {
                "success": True,
                "duplicate": True,
                "risk_score": current_status["risk_score"],
                "risk_level": current_status["risk_level"]
            }

        # 3단계: 즉시 recent_detections에 추가
        with detections_lock:
            recent_detections.append(data)
            detections_count = len(recent_detections)
            detections_list = list(recent_detections)

        logger.info(f"🗑️ 즉시 감지: {data.garbage_type} (신뢰도: {data.confidence:.2f}, 면적: {data.area}, 총 감지: {detections_count})")

        # 위험도 계산 전 로그
        logger.info(f"🔍 위험도 계산 시작 - 현재 감지 수: {detections_count}")

        # 4단계: 상태 업데이트
        with status_lock:
            previous_risk_score = current_status.get("risk_score", 0)
            previous_level = current_status.get("previous_level", "safe")

        update_status(detections_list)

        with status_lock:
            current_level = current_status["risk_level"]
            current_status["previous_level"] = current_level
            current_risk_score = current_status["risk_score"]

        # 유의미한 변화 확인 (더 민감하게)
        risk_change = abs(current_risk_score - previous_risk_score)
        significant_change = (risk_change >= 5.0) or (current_level != previous_level)  # 더 민감하게 조정

        # 알림 생성
        alert = None
        if current_level in ["warning", "caution", "danger"] and current_level != previous_level:
            with status_lock:
                blockage_info = current_status.get("blockage_percentage", 0)
                flow_restriction = current_status.get("flow_restriction", "알 수 없음")
                garbage_volume = current_status.get("garbage_volume", 0)
                risk_score_for_alert = current_status['risk_score']

            # 레벨별 메시지 구성
            if current_level == 'warning':
                emoji = '⚠️'
                level_text = '주의보'
            elif current_level == 'caution':
                emoji = '🟠'
                level_text = '경고'
            else:  # danger
                emoji = '🚨'
                level_text = '위험'

            detailed_message = (
                f"{emoji} 하수구 {level_text}!\n"
                f"• 막힘률: {blockage_info}%\n"
                f"• 흐름 제한: {flow_restriction}\n"
                f"• 축적 쓰레기량: {garbage_volume}cm³\n"
                f"• 위험도: {risk_score_for_alert:.2f}%"
            )

            alert = AlertData(
                level=current_level,
                message=detailed_message,
                timestamp=datetime.now(),
                risk_score=risk_score_for_alert
            )

            with detections_lock:
                recent_alerts.appendleft(alert)

            with status_lock:
                current_status["alerts_today"] += 1

            logger.warning(f"🚨 알림 발생: {detailed_message}")

        # 유의미한 변화시만 브로드캐스트
        if significant_change:
            with status_lock:
                status_copy = current_status.copy()

            broadcast_data = {
                "type": "detection",
                "data": data.model_dump(),
                "status": status_copy,
                "alert": alert.model_dump() if alert else None,
                "blockage_analysis": {
                    "blockage_percentage": status_copy.get("blockage_percentage", 0),
                    "garbage_volume": status_copy.get("garbage_volume", 0),
                    "flow_restriction": status_copy.get("flow_restriction", "알 수 없음"),
                    "accumulated_areas": status_copy.get("accumulated_areas", 0)
                },
                "ai_analysis": status_copy.get("ai_analysis", {})
            }

            await broadcast_to_clients(broadcast_data)

        # 위험도 계산 후 로그
        logger.info(f"📊 위험도 계산 완료 - 이전: {previous_risk_score:.2f}% → 현재: {current_risk_score:.2f}% (변화: {risk_change:.2f}%)")

        with status_lock:
            response_blockage = current_status.get("blockage_percentage", 0)
            response_flow = current_status.get("flow_restriction", "알 수 없음")

        return {
            "success": True,
            "duplicate": False,
            "significant_change": significant_change,
            "risk_score": current_risk_score,
            "risk_level": current_level,
            "blockage_percentage": response_blockage,
            "flow_restriction": response_flow,
            "alert_created": alert is not None
        }

    except Exception as e:
        logger.error(f"감지 처리 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """현재 시스템 상태 조회"""
    with status_lock:
        status_copy = current_status.copy()
    return StatusResponse(**status_copy)

@app.get("/blockage-analysis", response_model=BlockageAnalysis)
async def get_blockage_analysis():
    """현재 하수구 막힘 분석 결과"""
    with detections_lock:
        detections_list = list(recent_detections)
    return analyze_pipe_blockage(detections_list)

@app.get("/detections")
async def get_recent_detections(limit: int = 20):
    """최근 감지 기록 조회"""
    with detections_lock:
        detections = list(recent_detections)[-limit:]
        total = len(recent_detections)
    return {
        "detections": [d.model_dump() for d in detections],
        "total": total,
        "limit": limit
    }

@app.get("/alerts")
async def get_recent_alerts(limit: int = 10):
    """최근 알림 조회"""
    with detections_lock:
        alerts = list(recent_alerts)[:limit]
        total = len(recent_alerts)
    return {
        "alerts": [a.model_dump() for a in alerts],
        "total": total,
        "limit": limit
    }

@app.post("/reset")
async def reset_system():
    """시스템 상태 초기화"""
    global current_status, pending_detections

    with detections_lock:
        recent_detections.clear()
        recent_alerts.clear()
        pending_detections.clear()  # 대기 중인 감지들도 초기화
        confirmed_detections.clear()  # 확정된 감지들도 초기화

    with status_lock:
        current_status = {
            "risk_score": 0.0,
            "risk_level": "safe",
            "total_detections": 0,
            "last_detection": None,
            "alerts_today": 0,
            "pipe_status": "정상 - 원활한 흐름",
            "accumulation_rate": 0.0,
            "blockage_percentage": 0.0,
            "garbage_volume": 0.0,
            "flow_restriction": "없음",
            "accumulated_areas": 0
        }
        status_copy = current_status.copy()

    await broadcast_to_clients({
        "type": "reset",
        "status": status_copy
    })

    logger.info("🔄 시스템 초기화 완료 - 모든 감지 데이터 및 대기 상태 초기화")
    return {"success": True, "message": "시스템이 초기화되었습니다."}

# ==================== 비디오 스트리밍 ====================

from fastapi.responses import StreamingResponse

@app.get("/video_feed")
async def video_feed():
    """실시간 비디오 스트리밍"""
    def generate():
        while True:
            if current_frame is not None:
                ret, buffer = cv2.imencode('.jpg', current_frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                # 기본 이미지
                black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(black_frame, 'Camera not active', (200, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                ret, buffer = cv2.imencode('.jpg', black_frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            import time
            time.sleep(0.033)  # ~30 FPS

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/update_frame")
async def update_frame(frame_data: dict):
    """카메라 프레임 업데이트"""
    global current_frame, camera_active

    try:
        frame_base64 = frame_data.get('frame')
        if frame_base64:
            frame_bytes = base64.b64decode(frame_base64)
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is not None:
                current_frame = frame
                camera_active = True
                return {"success": True, "message": "Frame updated"}

        return {"success": False, "message": "Invalid frame data"}

    except Exception as e:
        logger.error(f"프레임 업데이트 오류: {e}")
        return {"success": False, "message": str(e)}

# ==================== WebSocket ====================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """실시간 데이터 스트리밍"""
    try:
        await websocket.accept()

        with clients_lock:
            connected_clients.append(websocket)
            client_count = len(connected_clients)

        logger.info(f"새 클라이언트 연결. 총 연결: {client_count}")

        # 연결 즉시 현재 상태 전송
        try:
            with status_lock:
                status_copy = current_status.copy()
                ai_analysis_copy = current_status.get("ai_analysis", {})

            with detections_lock:
                recent_detections_list = [d.model_dump() for d in list(recent_detections)[-5:]]
                recent_alerts_list = [a.model_dump() for a in list(recent_alerts)[:3]]

            initial_data = {
                "type": "initial",
                "status": status_copy,
                "recent_detections": recent_detections_list,
                "recent_alerts": recent_alerts_list,
                "ai_analysis": ai_analysis_copy
            }
            await websocket.send_text(json.dumps(initial_data, default=str))
            logger.info("초기 데이터 전송 완료")
        except Exception as e:
            logger.error(f"초기 데이터 전송 오류: {e}")

        # 연결 유지 루프
        while True:
            try:
                # 메시지 수신 (타임아웃 설정)
                message = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                
                if message == "ping":
                    await websocket.send_text("pong")
                    logger.debug("ping-pong 응답 완료")
                else:
                    logger.debug(f"알 수 없는 메시지 수신: {message}")
                    
            except asyncio.TimeoutError:
                # 30초 동안 메시지가 없으면 ping 전송
                try:
                    await websocket.send_text("ping")
                    logger.debug("서버에서 ping 전송")
                except:
                    logger.warning("ping 전송 실패 - 연결 끊어짐")
                    break
            except WebSocketDisconnect:
                logger.info("클라이언트가 연결을 끊었습니다")
                break
            except Exception as e:
                logger.error(f"WebSocket 메시지 처리 오류: {e}")
                break

    except WebSocketDisconnect:
        logger.info("WebSocket 연결 해제됨")
    except Exception as e:
        logger.error(f"WebSocket 연결 오류: {e}")
    finally:
        # 정리 작업
        with clients_lock:
            if websocket in connected_clients:
                connected_clients.remove(websocket)
            remaining_clients = len(connected_clients)
        logger.info(f"클라이언트 연결 정리 완료. 남은 연결: {remaining_clients}")

# ==================== 헬스체크 ====================

@app.get("/health")
async def health_check():
    """서버 상태 확인"""
    with clients_lock:
        client_count = len(connected_clients)

    with detections_lock:
        detection_count = len(recent_detections)

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "connected_clients": client_count,
        "camera_active": camera_active,
        "total_detections": detection_count,
        "version": "2.0.0"
    }

# ==================== 백그라운드 태스크 ====================

async def periodic_risk_update():
    """주기적으로 위험도 업데이트하여 자동 감소 처리"""
    while True:
        try:
            await asyncio.sleep(1.0)  # 1초마다 더 자주 실행

            # 오래된 감지 데이터 정리 (5초 이상 된 것들로 더 빠르게)
            now = datetime.now()
            old_detections = []

            with detections_lock:
                for detection in list(recent_detections):
                    try:
                        det_time = datetime.fromisoformat(detection.timestamp.replace('Z', '+00:00'))
                        seconds_ago = (now - det_time).total_seconds()
                        if seconds_ago > 5:  # 5초 이상 된 감지 (더 빠른 제거)
                            old_detections.append(detection)
                    except:
                        continue

                # 오래된 감지 제거
                for old_detection in old_detections:
                    if old_detection in recent_detections:
                        recent_detections.remove(old_detection)

                detections_list = list(recent_detections)

            # 주기적인 위험도 감소 (감지가 없어도 계속 감소)
            with status_lock:
                previous_risk = current_status.get("risk_score", 0)
                previous_level = current_status.get("risk_level", "safe")

            # 감지가 없고 위험도가 0보다 크면 자동 감소
            if not detections_list and previous_risk > 0:
                # 더 적극적인 감소율 적용 (1초마다 2% 감소)
                auto_decay_rate = 2.0
                new_risk = max(0.0, previous_risk - auto_decay_rate)

                with status_lock:
                    current_status["risk_score"] = new_risk
                    current_status["risk_level"] = get_risk_level(new_risk)
                    current_status["pipe_status"] = get_pipe_status(current_status["risk_level"])
                    status_copy = current_status.copy()

                # 변화가 있으면 브로드캐스트
                if new_risk != previous_risk:
                    broadcast_data = {
                        "type": "auto_decay",
                        "status": status_copy,
                        "message": f"쓰레기가 감지되지 않아 위험도가 자동으로 감소했습니다."
                    }
                    await broadcast_to_clients(broadcast_data)
                    logger.info(f"📉 자동 감소: {previous_risk:.1f}% → {new_risk:.1f}%")
            elif old_detections:
                # 오래된 감지 제거로 인한 재계산
                update_status(detections_list)

                with status_lock:
                    current_level = current_status["risk_level"]
                    new_risk = current_status["risk_score"]
                    status_copy = current_status.copy()

                # 위험도가 감소했거나 레벨이 변경되었을 때 브로드캐스트
                if new_risk < previous_risk or current_level != previous_level:
                    broadcast_data = {
                        "type": "detection_removal",
                        "status": status_copy,
                        "message": f"오래된 쓰레기 감지가 제거되어 위험도가 업데이트되었습니다. ({len(old_detections)}개 제거)"
                    }
                    await broadcast_to_clients(broadcast_data)
                    logger.info(f"📉 감지 제거로 위험도 업데이트: {previous_risk:.1f}% → {new_risk:.1f}% ({len(old_detections)}개 제거)")

        except Exception as e:
            logger.error(f"백그라운드 업데이트 오류: {e}")

# ==================== 애플리케이션 시작 이벤트 ====================

@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작시 백그라운드 태스크 시작"""
    logger.info("🚀 하수도 막힘 감지 시스템 시작")
    asyncio.create_task(periodic_risk_update())

# ==================== 디버깅을 위한 상세 로깅 추가 ====================

def log_risk_calculation_details(previous_risk: float, new_risk: float, detections_count: int):
    """위험도 계산 과정 상세 로깅"""
    logger.debug(f"위험도 계산 상세:")
    logger.debug(f"  - 이전 위험도: {previous_risk:.1f}%")
    logger.debug(f"  - 새 위험도: {new_risk:.1f}%")
    logger.debug(f"  - 변화량: {new_risk - previous_risk:+.1f}%")
    logger.debug(f"  - 감지 수: {detections_count}")
    logger.debug(f"  - 마지막 감지 후 경과시간: {get_time_since_last_detection():.1f}분")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")