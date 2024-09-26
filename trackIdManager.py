
class TrackIdManager:
    def __init__(self):
        self.next_id = 1  # 트랙 ID 초기값
        self.active_ids = set()  # 현재 활성화된 트랙 ID를 추적

    def get_new_id(self):
        new_id = self.next_id
        self.active_ids.add(new_id)
        self.next_id += 1
        return new_id

    def release_id(self, track_id):
        if track_id in self.active_ids:
            self.active_ids.remove(track_id)
