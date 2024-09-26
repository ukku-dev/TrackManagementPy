from enum import Enum
from globalVars import global_params

# 트랙 상태 Enum 정의
class TrackState(Enum):
    INIT = 0
    CONFIRMED = 1
    MISS = 2

class Track:
    def __init__(self, kalman_filter, track_id):
        self.kf = kalman_filter
        self.track_id = track_id  # 고유 트랙 ID
        self.age = 0
        self.total_visible_count = 0
        self.consecutive_invisible_count = 0
        self.state = TrackState.INIT  # 상태 초기화 (init)
        self.n_confirm = global_params['n_confirm']
        self.m_miss = global_params['m_miss']
        self.init_max_age = 2 * self.n_confirm  # init 상태에서 최대 유지할 수 있는 프레임 수
        self.last_measurement = None  # 마지막 매칭된 measurement 좌표

    def update(self, is_detected, measurement=None):
        self.age += 1

        if is_detected and measurement is not None:
            # 마지막 측정값을 저장
            self.last_measurement = measurement
            self.total_visible_count += 1
            self.consecutive_invisible_count = 0

            # INIT 상태에서 n_confirm 프레임 이상 감지되면 CONFIRMED로 전환
            if self.state == TrackState.INIT and self.total_visible_count >= self.n_confirm:
                self.state = TrackState.CONFIRMED

        else:
            self.consecutive_invisible_count += 1

            # MISS 상태로 전환
            if self.consecutive_invisible_count >= self.m_miss:
                self.state = TrackState.MISS

        # CONFIRMED 상태에서 감지되지 않으면 MISS 상태로 전환
        if self.state == TrackState.CONFIRMED and self.consecutive_invisible_count > 0:
            self.state = TrackState.MISS

    def get_display_position(self):
        # measurement가 있으면 measurement를 반환하고, 없으면 예측된 상태 반환
        if self.last_measurement is not None:
            return self.last_measurement
        else:
            state = self.kf.get_state()
            return state[0, 0], state[1, 0]
    def is_confirmed(self):
        return self.state == TrackState.CONFIRMED

    def is_lost(self):
        return self.state == TrackState.MISS

    def is_init(self):
        return self.state == TrackState.INIT

    # Init 상태에서 오래 유지되는 트랙을 제거
    def should_remove(self):
        return self.is_init() and self.age > self.init_max_age
