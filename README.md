# Traffic Vision Agent

面向夜间交通图片的视觉查询原型。将已有 YOLOv12 昼夜训练权重封装为检测、计数、最高置信度查询和可视化工具，再通过离线规则或 DeepSeek Function Calling 调用。用于本地演示与工程练习。

## 从仓库启动

需要 Windows、Python 3.11 和可用的网络来安装依赖。仓库包含推理权重 `weights/traffic_best.pt`；原始道路图片、标注数据集、API 密钥及运行产物不随仓库发布。首次安装：

```powershell
git clone https://github.com/YetLn/traffic-vision-agent.git
cd traffic-vision-agent
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\start.ps1
```

如果本机不支持 `py -3.11`，请使用已安装的 Python 3.11 创建 `.venv`。干净环境安装尚未单独复验；本机验证环境见文末。首次演示请自行上传有使用权限的道路图片。

## 本机启动

```powershell
cd D:\traffic-vision-agent
.\.venv\Scripts\python.exe app.py
```

浏览器打开 http://127.0.0.1:7860 ，上传道路图片，再依次询问：

1. 图里有什么交通标志？
2. 有多少个 ban？
3. 把它们框出来。
4. 哪个置信度最高？

也可运行 `start.ps1`。默认 CPU 推理，适配本机 MX350 2 GB 显存条件。首次加载比后续查询慢。

**如果页面启动时报 `startup-events ... failed (code 502)`**：这是本机系统代理（Windows Internet 设置里的 `127.0.0.1:7897`）拦截了 Gradio 对自身 localhost 的启动自检，进程本身已在本机端口监听。先设置环境变量再启动：

```powershell
$env:NO_PROXY = '127.0.0.1,localhost'; .\.venv\Scripts\python.exe app.py
```

命令行及测试：

```powershell
.\.venv\Scripts\python.exe main.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\verify_page.py            # 真实权重 + 页面 HTTP 自检
.\.venv\Scripts\python.exe scripts\verify_page.py --live-llm  # 追加一次真实 DeepSeek 问答
.\.venv\Scripts\python.exe scripts\verify_knowledge.py        # 知识库检索框架（离线）
.\.venv\Scripts\python.exe scripts\verify_context_compression.py  # 真实 API 8 轮压缩联调
```

命令行产生 `outputs/result.json` 与随机命名的标注图片。测试使用受控检测器及模拟 LLM，不消耗 API；真实图像与真实 API 验证单独记录于 `docs/validation.md`，报告在 `outputs/`。

## 模型选择与类别

仓库包含原研究训练得到的 `weights/traffic_best.pt`。权重 SHA256、来源说明与历史指标保存在 `weights/manifest.json`；训练数据未公开。

| 昼夜训练记录 | CSV 峰值 mAP50 | 同行 mAP50-95 |
| --- | ---: | ---: |
| YOLOv11-LGA | 0.64708 | 0.47604 |
| YOLOv12 | 0.63324 | 0.47287 |
| YOLOv13 | 0.65623 | 0.47291 |

这些是历史 CSV 中 mAP50-95 最大的一行，不是本项目重新评估的成绩，也未验证三次训练划分完全一致。选择 YOLOv12 是因为当前研究代码可加载并完成推理，避免在首版引入 LGA 或 HyperACE 等额外自定义结构。

**权重真实类别只有四类：`wran / ban / point-l / point-s`。** 保留原始拼写，不将训练标签改名。

类别语义依据项目自有的标注说明《labels instruction.docx》（团队内部依据，**非国标**；本项目未采用国标条款号作为依据）：

| 标签 | 含义 | 判定要点 |
| --- | --- | --- |
| `wran` | 警告标志 | 黄底黑边黑图案；非三角形但属警示牌面的也计入 |
| `ban` | 禁止标志 | 白底红圈红斜杠；红白箭头标志在统计口径上也计入 |
| `point-l` | 指路（方向）标志 | 蓝/绿/棕底，**含地名、方向箭头、距离、道路信息**，牌面较大 |
| `point-s` | 指示标志 | 蓝底白图案或绿底，圆形/矩形/方形，**不含**地名与距离，牌面较小 |

交叉验证：`scripts/inspect_label_sizes.py` 统计 1500 张图，`point-l` 标注框面积中位数 7636 px²，明显大于 `point-s` 2703、`ban` 2052、`wran` 1845，与上述“大牌指路 / 小牌指示”一致（报告 `outputs/label_size_report.json`）。

历史说明：早期标注方案还有第 5 类“白色辅助标志”（`traffic.yaml` 里的 `aux`）；现行夜间研究已不再使用，本项目的 checkpoint 与知识库都只覆盖上述四类。

没有限速值识别、OCR、遮挡分析或安全决策能力。文档中的五分类指早期标注方案，不代表本 checkpoint。

该 checkpoint 来源于研究 fork。`vendor/ultralytics` 固定了本机可用的原始实现，保留 AGPL-3.0 许可证；不能假设同版本号的 PyPI 包具有相同结构。应用启动优先导入 vendor。未修改外部科研仓库。

## DeepSeek 模式

```powershell
$env:DEEPSEEK_API_KEY = '填写你自己的密钥'
$env:DEEPSEEK_MODEL = 'deepseek-flash'
$env:DEEPSEEK_BASE_URL = 'https://api.deepseek.com'
.\.venv\Scripts\python.exe app.py
```

网页默认选择 `DeepSeek Tool Calling`。密钥可通过环境变量或本机 `.runtime/deepseek.json` 配置；进程环境变量优先。该文件被 Git 忽略，不要分享。模型名需匹配你的账户。`.env.example` 仅为变量模板，不自动加载。没有密钥时可选择离线模式演示。

DeepSeek 只接收问题、折叠后的文字上下文、近期对话及结构化检测结果，不发送图片像素或本地路径。工具名、参数字段、类别名均校验；最多 4 轮、每轮最多 4 个工具调用；请求超时 45 秒，SDK 最多重试一次。首次调用要求选择工具，工具结果随后回传模型，最终语言回答仍可能出现 LLM 错误，可展开轨迹核对。已验证路径：人工构造数据的真实两轮工具调用、8 轮上下文压缩联调，以及真实本机图片 `night_01.jpg` 的一次检测问答（发送内容仅问题、折叠上下文与类别/数量/置信度/边框，不含图片像素与本机路径）。报告见 `outputs/deepseek_validation.json`、`outputs/context_compression.json`、`outputs/page_verification.json`，记录见 `docs/validation.md`。

## 上下文压缩（Context Compression）

首版只是 `history[-12:]` 的窗口截断，会直接丢掉旧轮次。现在改成**确定性折叠**：

- 每轮问答后，历史超过 6 条即把较早轮次折叠进会话上下文 `state['context']`，只保留最近 4 条消息（即最近 2 轮）。
- 折叠只累积**统计信息**：图片名（内存图片用内容哈希短名，不泄露绝对路径）、首次检测类别统计、最近 3 条上下文条目（总数/类别/最高置信度）、当前关注类别、已折叠轮次。因此上下文体积与历史轮数和图片大小都无关。
- 同一张图、同一阈值只累计一次，所以重复追问不会把数量越加越大；换图或调阈值会清空上下文。
- 折叠后的 JSON 作为第二条 system 消息注入 DeepSeek 请求，并明确要求“摘要只用于解析指代，图片事实仍须调用工具”。
- 离线规则模式下，问“之前的对话摘要是什么？”会直接返回折叠上下文且不调用工具；问图片事实则必须调用工具，禁止用摘要编造检测结果。
- 页面“折叠后的会话上下文（Context Compression）”面板实时显示该上下文，便于演示与核对。

这不是语义摘要，也不调用 LLM 生成摘要：它是可复现、可测试的结构化折叠，避免模型在压缩阶段编造检测事实。

## 标志语义知识库（本地检索）

针对“这个标志是什么意思”，项目提供第 5 个工具 `lookup_sign`：

- `knowledge/signs.yaml`：四类的 含义 / 外观 / 典型场景 / 易混淆 / 依据来源，**已按项目标注说明填写**（`reference` 如实写“项目标注说明，非国标”）。
- `traffic_agent/knowledge.py`：本地 YAML + 关键词/别名检索，无向量库、无 embedding、无额外 API 依赖。
- 纪律：含义或依据为空的类别一律返回 `available=false`，Agent 只能回答“本地知识库缺少该类别依据”，**不得按外观或常识补写**；该工具也不能用来推断图中是否存在某类标志（真实联调中模型会主动声明“本次检测未检出该类别”）。
- 页面“标志语义知识库状态”面板显示已收录类别与依据；未收录类别会明确拒答而不是编一个答案。

这只是**关键词/别名检索**，不是向量检索。若后续替换为向量库并做 chunk 检索，才应表述为 RAG。验收：`scripts\verify_knowledge.py`（离线；`--live-llm` 追加真实 API 调用检查）。

## 指路牌文字读取与意图推断

`point-l`（class 2）里既有地名指示牌，也有大量**非地名提示牌**（行车规范、施工提醒、单位/景区提示等）。因此本项目不做“地名↔箭头”的强行解析，而是提供第 6 个工具 `read_sign_text`：

- **检测 → 原图分辨率裁剪 → 可读性判断 → OCR**。裁剪始终用原图，不用缩放到 640 的推理图；文字逐段保留 OCR 原文、置信度与四点坐标。
- **可读性门槛**：实测夜间样例检出的牌面裁剪只有 75×118、144×274 像素，肉眼都无法辨认；低于 160 px / 25600 px² 时直接报告“不具备可读条件”，**不让上层误以为牌面上没有文字**。
- **意图推断**：只有 DeepSeek 模式才做。提示词要求先逐段复述 OCR 原文，再把结论**标注为推测**并说明依据哪几段文字，低置信度文字要指出，文字太碎则直接说不确定；禁止给驾驶指令或安全结论。
- 离线规则模式只返回 OCR 原文并明确“未做含义推断”，不假装能推断。

实测（`scripts/verify_sign_intent.py --live-llm`，报告 `outputs/sign_intent_verification.json`）：模型逐段复述文字，把意图标注为推测并给出依据，还会主动声明“OCR 只给出文字内容，未给出各段文字在牌面上的方位排布，因此无法判断哪条路对应哪个方向”。

**历史限制**：旧版地名与方向解析没有通过验证。旧 20 牌评测中仅有 12 条路线，方向严格正确 4 条；上述文字召回只涵盖这 12 条路线，并非全部牌面文字。当前旧评测引用的裁剪名已经失效，评测脚本会报错并保留旧报告。2026-09-22 新增独立的证据定位试验入口，状态及原图结果见 `docs/direction_progress_20260922.md`，仍未达到方向准确率和覆盖率目标。

### 牌面文字分级（提示 / 设施 / 地名）

OCR 文字会进一步分级，供界面与模型区分“对驾驶员的要求”和“地点信息”：

| 级别 | 含义 | 示例 |
| --- | --- | --- |
| `prompt` | 对驾驶员的**行为要求** | 请按导向车道行驶、各行其道、禁止通行、减速慢行 |
| `info` | **设施/服务告知** | 地下停车场、游客中心、车辆管理分所 |
| `place` | 地名/道路名 | 平安路、嘉州大道 |
| `code` / `other` | 编号、距离、水印等 | G5513、500m、美篇 |

分级采用“强标记（请按/禁止/严禁/各行其道…）单独成立，泛指词（车辆/行车/车道…）需与强标记同现”的规则。留出集实测：单用泛指词会把「车辆管理分所」（单位名）与「公路自行车赛场」（赛事标识）误判为提示语，加上同现约束后精确率回到 1.00。

评测（`scripts/evaluate_prompt_classifier.py`，57 条人工判读，报告 `outputs/prompt_classifier_eval.json`）：prompt 精确率 1.00 / 召回 0.947，info 与 place 均为 1.00，整体准确率 0.982；唯一漏检是“注意行人”这类两字片段式提示语，已在评测集中如实标注为已知局限。

覆盖情况（`scripts/collect_prompt_signs.py`，541 + 241 张抽样 OCR）：`point-l` 中提示/规则类约占 **2%**，其余约 64% 为地名指路类、约 34% 为编号/距离/单位/待判。**该数据集里“施工/规范类标语牌”本就稀少**，且多以短句形式出现在指路牌上。端到端 8 块真实提示牌中 5 块成功抽出期望文字，其余 3 块因裁剪仅 61–200 px 被正确标记为“不具备可读条件”。

提示牌意图推断实测（`scripts/verify_prompt_intent.py`，真实 DeepSeek，报告 `outputs/prompt_intent_verification.json`）：问“这块牌想表达什么？”——提示/规则类牌面 **6/6** 判为“行车规范/提示”类并标注推测、引用依据；地名指路对照 **2/2** 未被说成驾驶员行为要求；不具备可读条件的牌面 **1/1** 明确拒答且不猜测。模型会主动区分「请进入集散车道」（提示语）、「S50 / 1.5km」（编号与距离）和「微信公」（水印式残缺文字，无法判断）。

**数据集注意事项**：2026-09-22 重新盘点发现，当前 Private 图片和原始标签的 stem 均唯一，未证实旧文档所称“跨批次同 ID、不同图片”。但存在不同文件名对应同一图片内容且标注框不同的情况。新提取器使用源图内容哈希与裁剪框共同命名，并保存标签来源、行号与哈希，避免裁剪覆盖；重复实体仍需人工复核。

## 简单指路牌方向解析（试验版，尚未验收）

页面增加“解析方向并显示证据”：道路原图 → 原图和补边两次 YOLO → 有 YOLO 支持的整牌边界恢复 → 原始分辨率裁剪 → OCR → 独立箭头形状 → 分行/共享箭头关联。输出原图坐标中的文字框、箭头框、关系、拒答原因，图中绘制对应证据。此入口完全本地运行。

当前支持部分独立分行布局和一箭头对应多个地名；第三轮加入保守的透视校正、OCR 框边缘重读与带 YOLO 证据的整牌边界恢复。35 张开发性质道路原图的 31 条可判定关系中，输出 9 条且均与当前近似标注一致，覆盖 29.0%；26 块负牌没有错误方向。但新增 10 张额外验证图的 5 条关系全部漏掉，**不能声称通过冻结验收**。新增 16 张原图可视标注（含 6 张夜间）和排除隐藏增强图的 v4 候选集已经保存；标注未经独立人审，100 个不同实体与 30/70 冻结划分尚未完成。详见 [第三轮改进与结果](docs/direction_upgrade_v3.md)；[第二轮](docs/direction_upgrade_v2.md)与[第一轮失败记录](docs/direction_progress_20260922.md)保留追溯。

```powershell
.\.venv\Scripts\python.exe scripts\read_directions.py --image D:\Private-dataset-master\val\images\08839.jpg --out outputs\direction_demo\08839
.\.venv\Scripts\python.exe scripts\verify_directions.py --gold outputs\direction_annotations_v1\gold.development.json --out outputs\direction_e2e_newrun
.\.venv\Scripts\python.exe scripts\evaluate_directions.py --gold outputs\direction_annotations_v1\gold.development.json --predictions outputs\direction_e2e_newrun\predictions.json --output outputs\direction_e2e_newrun\metrics.json
```

原图运行器不接受人工检测框。纯组件单元测试和裁剪诊断用于定位问题，不能替代原图评测。评测统计漏检、重复输出、负样本方向误报，并在没有输出关系时把准确率记为 `null`，不会记成 100%。

## 结构与工程边界

```text
图片 + 问题 → 会话状态（检测缓存 + 折叠上下文） → 离线路由 / DeepSeek 工具调用
                          ↓
             detect / count / highest_confidence / visualize
                          ↓
                YOLOv12 → JSON → 回答与标注图
```

- `traffic_agent/detector.py`：推理与绘制；共享模型使用锁串行推理。
- `traffic_agent/tools.py`：工具 schema、参数验证、单会话检测缓存、调用轨迹。
- `traffic_agent/agent.py`：离线多意图路由、有限 DeepSeek 循环、会话历史与上下文折叠。
- `traffic_agent/knowledge.py`：标志语义知识库与本地检索（缺依据拒答）。
- `knowledge/signs.yaml`：语义数据，依据项目标注说明填写。
- `app.py`：Gradio 页面；每个会话独立 State；只监听本机。
- `main.py`：可重复运行的图片检测入口。
- `scripts/`：真实 API、页面与知识库验收脚本；`inspect_label_sizes.py` 只读统计科研数据集标注框尺寸。
- `tests/`：换图失效、阈值失效、空检测、会话隔离、工具协议、上下文折叠、摘要注入、知识库拒答与工具接入等测试。

缓存键包含规范化图片内容、尺寸和阈值。同会话重复问答复用检测；换图或调阈值清空历史与折叠上下文。历史最多保留最近 2 轮，更早轮次折叠进结构化上下文。离线模式按明确关键词工作，不宣称是真正 LLM Agent。未实现向量检索、自进化、LangChain 或多模型路由。

`outputs/` 保存每次绘图，可按需手动清理。仓库仅提交推理权重，不提交道路原图、标注数据集、运行输出、密钥或虚拟环境。历史验证的摘要见 `reports/direction-v3-summary.json`；详细逐图结果需要原始数据，不能仅凭本仓库复现。发布示例前需自行确认素材使用权限。

## 环境复现

本机 `.venv` 使用 `D:\anaconda3\envs\pp\python.exe` 的 system-site-packages 复用已有 PyTorch 2.3.0，并在项目虚拟环境安装应用依赖。因此它不是可直接拷走的独立环境。其他机器请用 Python 3.11 建立干净虚拟环境后安装 `requirements.txt`，保留 vendor 与权重文件。干净环境安装尚需另外验证。

项目与内置研究代码的许可见 [LICENSE](LICENSE) 和 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

参考：[Ultralytics YOLO12 文档](https://docs.ultralytics.com/models/yolo12/)、[DeepSeek 工具调用文档](https://api-docs.deepseek.com/guides/tool_calls)。API 与运行依赖以本机验收记录为准。
