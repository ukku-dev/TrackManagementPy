import numpy as np
from scipy.spatial import KDTree
from globalVars import global_params
from kalmanFilter import KalmanFilter
from track import Track
from trackIdManager import TrackIdManager
import time

class TrackManager:
    def __init__(self):
        self.tracks = []
        self.nds_distance = global_params['nds_distance']
        self.id_manager = TrackIdManager()
        self.track_positions = None
        self.tree = None

    def update_tracks(self, measurements):
        unmatched_measurements = []
        matched_tracks = set()

        # 1. 기존 트랙 예측
        start_time = time.time()
        for track in self.tracks:
            track.kf.predict()
        print(f"Step 1 - Track Prediction Time: {(time.time() - start_time) * 1000:.2f} ms")

        # 2. 트랙 위치 배열 생성 및 KDTree 생성 (필요 시에만)
        if len(self.tracks) > 0:
            new_track_positions = np.array([track.kf.get_state()[:2].flatten() for track in self.tracks])
            
            # 트랙 위치가 크게 변한 경우에만 KDTree 재생성
            if self.track_positions is None or not np.array_equal(self.track_positions, new_track_positions):
                self.track_positions = new_track_positions
                self.tree = KDTree(self.track_positions)
                print("KDTree has been updated due to track position changes.")

            # 3. KDTree로 가장 가까운 트랙 찾기
            start_time = time.time()
            measurement_positions = np.array(measurements)
            distances, indices = self.tree.query(measurement_positions, distance_upper_bound=self.nds_distance)
            print(f"Step 2 - KDTree Distance Calculation Time: {(time.time() - start_time) * 1000:.2f} ms")
            
            # 4. 각 측정값에 대해 가장 가까운 트랙 찾기
            start_time = time.time()
            for j, measurement in enumerate(measurements):
                min_distance = distances[j]
                min_track_index = indices[j]

                if min_distance < self.nds_distance:
                    closest_track = self.tracks[min_track_index]
                    closest_track.kf.update(np.matrix(measurement).T)
                    closest_track.update(True, measurement)
                    matched_tracks.add(closest_track)
                else:
                    unmatched_measurements.append(measurement)
            print(f"Step 3 - Track Matching Time: {(time.time() - start_time) * 1000:.2f} ms")
        else:
            unmatched_measurements = measurements

        # 5. 매칭되지 않은 트랙 상태 업데이트
        start_time = time.time()
        for track in self.tracks:
            if track not in matched_tracks:
                track.update(False)
        print(f"Step 4 - Unmatched Track Update Time: {(time.time() - start_time) * 1000:.2f} ms")

        # 6. 새로운 트랙 추가
        start_time = time.time()
        for x, y in unmatched_measurements:
            new_kf = KalmanFilter()
            new_kf.update(np.matrix([x, y]).T)
            new_track_id = self.id_manager.get_new_id()
            new_track = Track(new_kf, new_track_id)
            new_track.last_measurement = (x, y)
            self.tracks.append(new_track)
        print(f"Step 5 - New Track Addition Time: {(time.time() - start_time) * 1000:.2f} ms")

        # 7. 손실된 트랙 제거
        start_time = time.time()
        self.tracks = [track for track in self.tracks if not track.is_lost() and not track.should_remove()]
        print(f"Step 6 - Lost Track Removal Time: {(time.time() - start_time) * 1000:.2f} ms")

    def get_top_k_confirmed_tracks(self, k):
        confirmed_tracks = [track for track in self.tracks if track.is_confirmed()]
        confirmed_tracks.sort(key=lambda t: t.age)
        return confirmed_tracks#[:k]

    def get_tracks(self):
        return [(track.kf.get_state(), track.is_confirmed(), track.is_lost(), track.track_id) for track in self.tracks]
