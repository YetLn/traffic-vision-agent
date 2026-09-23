# DeepSeek 接口迁移验收

接口为 https://api.deepseek.com ，模型为 deepseek-flash，关闭思考模式以适配当前有界工具循环。配置从 Git 忽略的 `.runtime/deepseek.json` 加载，环境变量可以覆盖；密钥不写入源码或报告。

已完成：

- 替换 Qwen 客户端配置、函数命名、界面模式与使用说明。
- 页面默认 DeepSeek Tool Calling，保留离线规则模式。
- 8 项自动测试通过。
- 使用真实 API、人工构造的两条 ban 检测记录验证两轮问答：第一轮调用 count，第二轮正确将“它们”解析为 ban 并调用 visualize；第二轮缓存命中。
- 返回“检测结果：ban 类别共 2 个。”及“已绘制 ban 类别的检测框，共 2 个。”。

完整无密钥报告位于 `outputs/deepseek_validation.json`。运行 `python scripts/verify_deepseek.py` 可重复此真实 API 测试，会产生 API 用量，默认不读取本机图片或权重。

本次真实 API 测试只验证接口、模型路由、结果回传与多轮上下文；可视化测试工具仅保存空白测试图片，不用于证明 YOLO 绘图正确。YOLO 检测与绘图已在上一版独立验收。

真实本机图片的检测 JSON 外发测试被自动审批拒绝，尚未执行成功。审批要求明确授权向 DeepSeek 发送测试问句及本机图片产生的类别、数量、置信度和边框；原图无需发送。获得授权后可使用 `--real-images` 完成该项。
