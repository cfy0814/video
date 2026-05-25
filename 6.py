# 先安装依赖：pip install -U langchain-openai opencv-python
import cv2
import base64
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# ===== 1. 配置API（确保用的是VL视觉模型）=====
chat_model = ChatOpenAI(
    openai_api_key="sk-4e8c9510a00d4284ae27e0cd3f1542c6",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen3-vl-plus",  # 阿里云可用的多模态模型，小写也可以
    temperature=0.7,
    max_tokens=1024
)

# ===== 2. 把视频帧直接转成base64（不用存本地图片）=====
def frame_to_base64(frame):
    # 将OpenCV的帧编码为JPG格式
    ret, buffer = cv2.imencode('.jpg', frame)
    if not ret:
        return None
    img_bytes = buffer.tobytes()
    # 拼接成API需要的格式
    return f"data:image/jpeg;base64,{base64.b64encode(img_bytes).decode('utf-8')}"

# ===== 3. 调用大模型分析单帧画面 =====
def analyze_frame(frame, prompt="描述这张图片的内容"):
    img_b64 = frame_to_base64(frame)
    if not img_b64:
        return "帧编码失败"
    
    messages = [
        HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": img_b64}}
            ]
        )
    ]
    try:
        response = chat_model.invoke(messages)
        return response.content
    except Exception as e:
        return f"API调用失败: {str(e)}"

# ===== 4. 读取视频流（先测试本地视频）=====
if __name__ == "__main__":
    # 先测试本地视频文件，确保路径正确
    import os
    print("当前目录文件列表：", os.listdir("."))  # 打印文件，确认2.mp4存在
    video_path = "./2.mp4"
    if not os.path.exists(video_path):
        print(f"❌ 文件不存在：{video_path}")
        exit()

    video_source = video_path
    analyze_interval = 30
    frame_count = 0

    # 打开视频流
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("❌ 无法打开视频流，请检查地址/文件路径")
        exit()

    print("✅ 视频流已打开，按 'q' 退出程序")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("📹 视频流读取结束")
            break

        frame_count += 1
        # 每隔N帧调用一次大模型
        if frame_count % analyze_interval == 0:
            print(f"\n===== 第 {frame_count} 帧 AI 分析结果 =====")
            result = analyze_frame(frame)
            print(result)

        # 显示视频画面
        cv2.imshow("Video Stream", frame)
        # 按q键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()