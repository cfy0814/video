import cv2
import subprocess
import numpy as np
from ultralytics import YOLO

# 1. 加载 YOLO 模型（本地模型文件，避免下载失败）
model = YOLO("yolov8n-pose.pt") 

# 流地址
rtsp_input_url = "rtsp://127.0.0.1:25544/input"
rtsp_output_url = "rtsp://127.0.0.1:25544/output"

# 只打开一次视频流！
cap = cv2.VideoCapture(rtsp_input_url)
if not cap.isOpened():
    print(f"❌ 无法打开视频流：{rtsp_input_url}")
    exit()

# 获取视频信息
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
print(f"✅ 视频流已打开：{width}x{height} @ {fps}fps")

# 2. FFmpeg 推流命令（彩色BGR24专用）
ffmpeg_cmd = [
    'ffmpeg',
    '-y',
    '-f', 'rawvideo',
    '-vcodec', 'rawvideo',
    '-pix_fmt', 'bgr24',
    '-s', f'{width}x{height}',
    '-r', str(fps),
    '-i', '-',
    '-c:v', 'libx264',
    '-pix_fmt', 'yuv420p',
    '-preset', 'ultrafast',
    '-tune', 'zerolatency',
    '-crf', '23',
    '-threads', '4',
    '-f', 'rtsp',
    '-rtsp_transport', 'tcp',
    rtsp_output_url
]

process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
expected_len = width * height * 3

# 3. 循环处理（加入断流重连逻辑）
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️  读取帧失败，尝试重新连接视频流...")
        cap.release()
        cap = cv2.VideoCapture(rtsp_input_url)
        if not cap.isOpened():
            print("❌ 重连失败，退出程序")
            break
        continue  # 重连成功，跳过本次循环，重新读取帧

    # YOLO 姿态检测
    results = model(frame, stream=True, verbose=False)
    annotated_frame = frame.copy()
    for r in results:
        annotated_frame = r.plot()

    # 数据长度校验，避免FFmpeg解码错误
    actual_len = len(annotated_frame.tobytes())
    if actual_len != expected_len:
        print(f"⚠️  数据长度异常：预期 {expected_len}，实际 {actual_len}，跳过")
        continue

    # 写入FFmpeg
    try:
        process.stdin.write(annotated_frame.tobytes())
    except BrokenPipeError:
        print("❌ FFmpeg进程已关闭，退出程序")
        break

# 清理资源
cap.release()
cv2.destroyAllWindows()
if process.poll() is None:
    process.stdin.close()
    process.wait()