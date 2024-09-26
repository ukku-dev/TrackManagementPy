import cv2
import numpy as np
import random

# 랜덤 타겟 생성 및 움직임 시뮬레이션 (복잡한 배경의 그레이스케일 영상)
def generate_moving_targets_video(output_filename, width, height, num_targets, num_frames, frame_rate):
    # 비디오 생성 (그레이스케일)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_filename, fourcc, frame_rate, (width, height), isColor=False)

    # 타겟 초기 위치 및 속도 설정
    targets = []
    for _ in range(num_targets):
        # 타겟 초기 위치 (x, y), 속도 (vx, vy)
        x = random.randint(50, width - 50)
        y = random.randint(50, height - 50)
        vx = random.uniform(-5, 5)  # x축 속도
        vy = random.uniform(-5, 5)  # y축 속도
        targets.append([x, y, vx, vy])

    for frame_idx in range(num_frames):
        # 빈 프레임 생성 (노이즈 추가된 그레이스케일 배경)
        frame = np.random.randint(0, 50, (height, width), dtype=np.uint8)  # 약간의 노이즈 배경

        # 랜덤 선과 패턴 추가
        if frame_idx % 10 == 0:  # 프레임마다 랜덤 선 그리기
            for _ in range(10):  # 랜덤한 직선 추가
                x1, y1 = random.randint(0, width), random.randint(0, height)
                x2, y2 = random.randint(0, width), random.randint(0, height)
                cv2.line(frame, (x1, y1), (x2, y2), random.randint(50, 150), 1)
        
        if frame_idx % 15 == 0:  # 프레임마다 랜덤 원 추가
            for _ in range(5):  # 랜덤한 원 추가
                x, y = random.randint(0, width), random.randint(0, height)
                radius = random.randint(10, 50)
                cv2.circle(frame, (x, y), radius, random.randint(50, 150), 1)

        for target in targets:
            # 타겟 위치 업데이트
            target[0] += target[2]  # x 좌표 업데이트
            target[1] += target[3]  # y 좌표 업데이트

            # 화면 경계에서 반사
            if target[0] <= 20 or target[0] >= width - 20:
                target[2] *= -1  # x축 속도 반전
            if target[1] <= 20 or target[1] >= height - 20:
                target[3] *= -1  # y축 속도 반전

            # 타겟 그리기 (원의 크기: 20, 밝기: 255)
            cv2.circle(frame, (int(target[0]), int(target[1])), 1, 255, -1)

        # 프레임에 타겟을 그린 후 비디오에 추가
        out.write(frame)

    out.release()
    print(f"'{output_filename}' 생성 완료.")

# 랜덤 타겟 동영상 생성
if __name__ == "__main__":
    output_filename = 'moving_targets_complex_background.avi'
    video_width = 3840
    video_height = 480
    num_targets = 20  # 움직이는 타겟 개수
    num_frames = 300  # 총 프레임 수
    frame_rate = 20  # FPS

    generate_moving_targets_video(output_filename, video_width, video_height, num_targets, num_frames, frame_rate)
