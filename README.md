# 视频流实时AI分析项目

基于OpenCV + 大语言模型的视频流实时多模态分析工具，支持本地视频/RTSP流抽帧识别，可对画面内容进行文字描述与场景理解。

---

## ✨ 项目功能

- **视频流读取**：支持本地视频文件、RTSP实时流拉取
- **实时抽帧**：可自定义抽帧间隔，避免调用大模型过于频繁
- **多模态分析**：接入大语言模型，对视频帧进行内容识别与描述
- **可视化预览**：OpenCV实时播放视频画面，按q键可退出程序

---

## 🛠️ 环境依赖

### 1. 安装依赖包

```bash
pip install -U langchain-openai opencv-python langchain-core
```

### 2. 项目结构

```
real/
├── 6.py                # 主程序：视频流实时AI分析+YOLO检测
├── 2.mp4               # 示例视频文件
├── yo.pt          # YOLO预训练权重（首次运行自动下载）
├── EasyDarwin/         # RTSP流服务器（可选）
├── ffmpeg/             # 推流工具（可选）
└── README.md           # 项目说明文档
```

------

## YOLO 目标检测
1.功能说明
兼容本地视频 / RTSP 实时流逐帧目标检测
自动绘制检测框、类别标签、置信度
可与多模态大模型联动：先 YOLO 检测物体，再由 VL 大模型做场景语义理解
支持自定义置信度阈值、检测类别、GPU 加速推理

2.使用方式
安装依赖 pip install ultralytics
无需手动下载权重，首次运行自动拉取 yolov8n.pt
保持原有本地视频 / RTSP 流启动方式不变，直接运行 python 6.py
视频窗口自动叠加目标检测框，终端同时输出大模型场景描述

3.联动大模型逻辑
YOLO 先识别画面中人、车、物品等目标
筛选关键检测帧转为 Base64
传入 qwen3-vl-plus多模态大模型，结合检测结果做深度场景解读
可扩展实现异常目标报警、人数统计、区域入侵检测

4.YOLO 常见问题
权重下载失败：手动下载 yolov8n.pt 放入项目根目录
检测卡顿：调低 imgsz、提高 conf 阈值，或启用 GPU
RTSP 流检测延迟：设置 YOLO stream=True 流式推理优化
检测类别不符：可加载自定义训练 YOLO 权重，适配业务场景

## 🚀 快速运行

### 方式 1：本地视频文件分析

修改6.py中的视频源为本地文件路径：

```
video_source = "./2.mp4"  # 替换为你的视频文件路径
```

运行程序：

```
python 6.py
```

### 方式 2：RTSP 实时流分析

1. 启动 EasyDarwin 服务

2. 使用 ffmpeg 推流：

   ```
   ffmpeg -re -stream_loop -1 -i 2.mp4 -c copy -f rtsp rtsp://127.0.0.1:25544/input
   ```

3. 修改 6.py 中的视频源为 RTSP 地址：

   ```
   video_source = "rtsp://127.0.0.1:25544/input"
   ```

4. 运行程序：

   ```
   python 6.py
   ```

------

## 📝 代码核心说明

### 1. 大模型配置

```
chat_model = ChatOpenAI(
    openai_api_key="你的API密钥",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen3-vl-plus",  # 多模态视觉模型
    temperature=0.7,
    max_tokens=1024
)
```

### 2. 帧转 Base64（无本地文件）

直接将 OpenCV 读取的视频帧转为 API 需要的 Base64 格式，无需保存图片文件：

```
def frame_to_base64(frame):
    ret, buffer = cv2.imencode('.jpg', frame)
    if not ret:
        return None
    img_bytes = buffer.tobytes()
    return f"data:image/jpeg;base64,{base64.b64encode(img_bytes).decode('utf-8')}"
```

### 3. 自定义分析频率

修改analyze_interval控制抽帧间隔，数值越小分析越频繁：

```
analyze_interval = 30  # 每30帧调用一次大模型
```

------

## 📸 运行效果

程序运行后，会同时显示：

1. OpenCV 窗口：实时播放视频画面
2. 终端输出：大模型对每帧画面的详细分析结果（如人物动作、服装、场景描述等）
<img width="1521" height="610" alt="屏幕截图 2026-05-25 102848" src="https://github.com/user-attachments/assets/dc705f57-668c-4a04-a560-f272613c518b" />

------

## ⚠️ 常见问题

### 1. 无法打开视频流

- 检查视频文件路径是否正确，确保文件未损坏
- 若使用 RTSP 流，确认 EasyDarwin 已启动且推流成功
- 尝试用 VLC 播放器打开 RTSP 地址，测试流是否正常

### 2. API 调用失败

- 检查 API 密钥、base_url 和模型名称是否正确
- 确保使用的是支持多模态的 VL 模型（如 qwen3-vl-plus）
- 确认账户余额充足，未触发 API 限流

### 3. 终端标黄 / 报错

- 激活你的虚拟环境
- 重新安装依赖包 `pip install -U langchain-openai opencv-python`

------

## 📌 后续扩展方向

- 接入目标检测模型（如 YOLO），实现画面中物体的自动标注
- 增加异常行为检测功能，对特定动作 / 场景进行预警
- 优化抽帧逻辑，支持关键帧智能筛选，减少 API 调用次数
- 增加结果保存功能，将分析日志导出为文件
