import cv2
import numpy as np
import time
from globalVars import global_params
from trackManager import TrackManager

def main():
    # Blob Detector 설정
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = global_params['min_blob_area']  # 작은 블롭도 탐지할 수 있도록 최소 영역 크기 설정
    params.maxArea = global_params['max_blob_area']  # 최대 영역 크기 설정
    params.filterByCircularity = True
    params.minCircularity = global_params['blob_circularity']  # 원형성 기준
    params.filterByColor = True
    params.blobColor = 255  # 흰색 블롭 탐지

    detector = cv2.SimpleBlobDetector_create(params)

    # 칼만 필터 초기화
    track_manager = TrackManager()

    # 비디오 불러오기
    cap = cv2.VideoCapture(global_params['input_video'])

    # 영상 크기 정보 얻기
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # 출력 영상 설정
    out = cv2.VideoWriter(global_params['output_video'], cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), global_params['frame_rate'], (frame_width, frame_height), isColor=False)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # 영상이 끝났을 때, 다시 처음으로 돌아감
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        # 1. Blob Detection 시간 측정
        start_blob = time.time()

        # 그레이스케일 영상으로 변환
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 이진화 처리 (threshold 설정)
        _, thresh_frame = cv2.threshold(gray_frame, global_params['threshold_value'], 255, cv2.THRESH_BINARY)
        # contour
        keypoints = []
        contours, _ = cv2.findContours(thresh_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        max_objects=50
        size_threshold=1
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # 중심 좌표와 너비, 높이를 사용하여 KeyPoint 생성 (중심점과 크기)
            center_x = x + w / 2
            center_y = y + h / 2
            size = max(w, h)  # 객체의 크기를 size로 설정
            # 크기 기준 필터링 (size_threshold 이상만 추가)
            if size >= size_threshold:
                keypoints.append((center_x, center_y, size))

        # 이미지 중심으로부터 가까운 순서로 정렬
        image_center = (frame.shape[1] / 2, frame.shape[0] / 2)
        keypoints.sort(key=lambda kp: np.hypot(kp[0] - image_center[0], kp[1] - image_center[1]))

        # 상위 max_objects 개수만 선택
        keypoints = keypoints[:max_objects]

        # Keypoints로부터 measurements 생성
        measurements = [(kp[0], kp[1]) for kp in keypoints]
        
        end_blob = time.time()
        blob_time = (end_blob - start_blob)*1000.

        # Blob 좌표 저장
        # measurements = [(kp.pt[0], kp.pt[1]) for kp in keypoints]
        print(len(measurements))

        # 2. 칼만 필터 트랙 업데이트 시간 측정
        start_kalman = time.time()
        # 트랙 업데이트
        track_manager.update_tracks(measurements)
        
        end_kalman = time.time()
        kalman_time = (end_kalman - start_kalman)*1000.

        # 트랙 상태 그리기
        # for track in track_manager.tracks:
        #     state = track.kf.get_state()  # 칼만 필터의 상태 추정치
        #     x, y = int(state[0]), int(state[1])
        #     if track.is_init():
        #         color = (255, 255, 0)  # 노란색으로 초기 상태를 표시
        #     elif track.is_confirmed():
        #         color = (0, 255, 0)  # 초록색으로 confirm 상태 표시
        #     else:
        #         color = (0, 0, 255)  # 빨간색으로 miss 상태 표시
        #     cv2.rectangle(frame, (x - 20, y - 20), (x + 20, y + 20), color, 2)
        
        for meas in measurements:
            x, y = meas
            x, y = int(x), int(y)
            
            color = (127, 0, 127)  # 노란색으로 초기 상태를 표시

            cv2.rectangle(frame, (x - 10, y - 10), (x + 10, y + 10), color, 1)
        
        # 트랙 상태 그리기 (먼저 생성된 confirm 상태 트랙 중 K개만 표시)
        top_k_tracks = track_manager.get_top_k_confirmed_tracks(global_params['display_target_num'])

        for track in top_k_tracks:
            x, y = track.get_display_position()
            x, y = int(x), int(y)
            
            if track.is_init():
                color = (255, 255, 0)  # 노란색으로 초기 상태를 표시
            elif track.is_confirmed():
                color = (0, 255, 0)  # 초록색으로 confirm 상태 표시
            else:
                color = (0, 0, 255)  # 빨간색으로 miss 상태 표시
            cv2.rectangle(frame, (x - 20, y - 20), (x + 20, y + 20), color, 2)
            cv2.putText(frame, f"ID: {track.track_id}", (x - 20, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            

        # 결과 영상 출력
        # out.write(frame)
        cv2.imshow('Blob Detection', frame)
        cv2.imshow('thresh_frame Detection', thresh_frame)
        
        # 각 처리 단계별 시간 출력
        print(f"Blob Detection Time: {blob_time:.6f} ms, Kalman Update Time: {kalman_time:.6f} ms")


        if cv2.waitKey(30) & 0xFF == ord('q'):
            break


    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
