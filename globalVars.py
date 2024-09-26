

# 글로벌 파라미터 관리 클래스
global_params = {
    'dt': 1/20,  # time step
    'u_x': 0,  # x 방향 가속도
    'u_y': 0,  # y 방향 가속도
    'std_acc': 0.5,  # 프로세스 노이즈 표준편차
    'x_std_meas': 0.05,  # x 좌표 측정 노이즈 표준편차
    'y_std_meas': 0.05,  # y 좌표 측정 노이즈 표준편차
    'n_confirm': 3,  # 확인 상태로 전환되는 프레임 수
    'm_miss': 1,  # 손실 상태로 전환되는 프레임 수
    'nds_distance': 30,  # NDS 거리
    'min_blob_area': 1,  # Blob Detection 최소 영역
    'max_blob_area': 100,  # Blob Detection 최대 영역
    'blob_circularity': 0.7,  # 원형성 기준
    'threshold_value': 127,  # 이진화 기준값
    #'input_video': 'moving_targets_complex_background.avi',  # 입력 영상 파일
    'input_video': 'moving_targets_gray.avi',
    'output_video': 'output_blob_detection.avi',  # 출력 영상 파일
    'frame_rate': 20,  # FPS
    'display_target_num' : 30,
}
