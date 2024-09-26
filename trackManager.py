import numpy as np
from globalVars import global_params
from track import Track
from kalamanFilter import KalmanFilter
from trackIdManager import TrackIdManager

class TrackManager:
    def __init__(self):
        self.tracks = []
        self.nds_distance = global_params['nds_distance']
        self.id_manager = TrackIdManager()  # ID 관리자를 생성

    def update_tracks(self, measurements):
        unmatched_measurements = []  # 매칭되지 않은 측정값

        # 1. 기존 트랙 예측
        for track in self.tracks:
            track.kf.predict()

        # 2. 측정값과 기존 트랙 비교
        matched_tracks = set()  # 이미 매칭된 트랙

        for measurement in measurements:
            x, y = measurement
            matched = False
            for track in self.tracks:
                # NDS 거리 내에 있는지 확인
                distance = np.linalg.norm(track.kf.get_state()[:2] - np.array([x, y]).reshape(-1, 1)[:2])
                if distance < self.nds_distance:
                    track.kf.update(np.matrix([x, y]).T)
                    track.update(True, (x, y))  # 매칭된 경우 measurement를 업데이트
                    matched_tracks.add(track)  # 매칭된 트랙 기록
                    matched = True
                    break

            if not matched:
                unmatched_measurements.append((x, y))

        # 3. 매칭되지 않은 트랙에 대해 업데이트 (miss 상태로 전환)
        for track in self.tracks:
            if track not in matched_tracks:
                track.update(False)  # 매칭되지 않은 트랙은 False로 업데이트

        # 4. 매칭되지 않은 측정값에 대해 새로운 트랙 추가
        for x, y in unmatched_measurements:
            new_kf = KalmanFilter()
            new_kf.update(np.matrix([x, y]).T)
            new_track_id = self.id_manager.get_new_id()  # 새로운 트랙 ID 생성
            new_track = Track(new_kf, new_track_id)
            new_track.last_measurement = (x, y)  # 생성된 트랙의 초기 위치 설정
            self.tracks.append(new_track)

        # 5. 손실된 트랙 및 init 상태로 오래 유지된 트랙 제거
        lost_tracks = [track for track in self.tracks if track.is_lost() or track.should_remove()]
        for track in lost_tracks:
            self.id_manager.release_id(track.track_id)  # 트랙이 제거될 때 ID 해제
        self.tracks = [track for track in self.tracks if not track.is_lost() and not track.should_remove()]

    # confirm 상태인 트랙 중에서 K개만 선택
    def get_top_k_confirmed_tracks(self, k):
        confirmed_tracks = [track for track in self.tracks if track.is_confirmed()]
        confirmed_tracks.sort(key=lambda t: t.age)  # 생성된 순서대로 정렬 (나이순)
        return confirmed_tracks[:k]

    def get_tracks(self):
        return [(track.kf.get_state(), track.is_confirmed(), track.is_lost(), track.track_id) for track in self.tracks]
