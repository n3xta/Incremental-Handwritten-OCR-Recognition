### 系统概览
- 核心引擎: Google Cloud Vision `DOCUMENT_TEXT_DETECTION`（API key REST 或 客户端库 + ADC 自动切换），支持自定义端点。
  - 参考文档: [Google Cloud Vision - Detect handwriting in images](https://cloud.google.com/vision/docs/handwriting)
- 双模式运行:
  - GUI 模式：摄像头取流，绘制 `text_annotations` 框，实时显示全文。
  - Console 模式：仅打印“稳定增量行”，并在空闲后汇总“(send) …”。

### OCR 调度
- **采样节流**: `ocr_interval = 0.7s`，降低费用与抖动。
- **输入源**: OpenCV 摄像头帧，JPEG 编码再请求 OCR。

### 文本清洗与合并
- **去噪与规范化**:
  - 去除多种项目符号、花式引号、收敛多空格。
  - 统一省略号成 `...`；句尾 `?!` 串若同时含 `?` 与 `!`，归一为 ` ??!`。
- **行合并**: 短前缀行、小写续行、前一行无终止标点、符号起始的下一行会与上一行合并，缓解换行切分抖动。

### 稳定器（增量稳定输出）
- **近似合并**: 使用相似度（`difflib`）将轻微变体归并为同一“键”。
- **分数模型**: 命中加分（`boost`），逐帧衰减（`decay`），达到阈值（`threshold`）才视作“稳定行”输出一次。
- **冷却时间**: `cooldown_s`，同一稳定句在冷却期内不重复触发；避免“(send) 后立刻又稳定”的回弹。
- **去重策略**: 批内/段内按规范化键去重，避免“Guys... / Guys....”等并列重复。

### 空闲发送（(send)）
- **触发条件**: 自最后一次新“稳定行”出现后，使用 monotonic 计时达到 `idle_timeout_s`（默认 2.5s）即输出一条聚合“(send) …”，随后清空段内缓冲，不重置稳定器（依靠冷却控制重复）。
- **聚合内容**: 段内所有稳定行，规范化去重后拼接为一行；可改为输出 JSON 事件，便于前端解析。

### 关键参数（可调）
- **采样**: `ocr_interval=0.7`
- **稳定器**: `threshold=2.0`、`decay=0.85`、`fuzzy=0.92`、`cooldown_s=6.0`
- **空闲发送**: `idle_timeout_s=2.5`

### 环境变量
- `GOOGLE_CLOUD_API_KEY`（或 `GCP_API_KEY` / `VISION_API_KEY`）: 仅有 API key 时走 REST。
- `GOOGLE_VISION_API_ENDPOINT`: 自定义端点（如 EU 区域）。
- `BACKEND_IDLE_SEND_S`: 空闲发送阈值（秒）。
- `BACKEND_COOLDOWN_S`: 稳定句冷却（秒）。

### 运行方式
- GUI:
```bash
python backend/main.py
```
- Console（推荐对接前端）:
```bash
python backend/main.py --console
```

- 如果需要结构化输出，打印处可切换为:
```python
print(json.dumps({"type": "stable_text", "text": s}), flush=True)
# 或在发送时:
print(json.dumps({"type": "send", "text": send_text}), flush=True)
```

- **结果**: 控制台只会输出跨帧稳定后的“新”句子；在约 2.5s 无新增后，输出一条聚合“(send) …”。通过冷却与去重避免重复与拼接噪声。