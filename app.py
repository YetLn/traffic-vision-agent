from traffic_agent.config import ROOT, OUTPUTS
import gradio as gr
from traffic_agent.agent import ask, context_view, new_state
from traffic_agent.detector import TrafficSignDetector
from traffic_agent.knowledge import inventory
from traffic_agent.direction_reader import read_directions, draw_evidence, summarize_directions
import uuid


def knowledge_status():
    data = inventory()
    rows = '\n'.join(
        f"- `{row['class_name']}`：{'已收录，依据 ' + row['reference'] if row['available'] else '待补充依据（查询会明确拒答）'}"
        for row in data['classes'])
    return (f'### 标志语义知识库（本地检索，非向量库）\n'
            f'已收录 {data["available_count"]}/{data["total"]} 类。{rows}\n\n'
            '含义问题只引用知识库原文；没有依据的类别明确拒答，不凭外观推测。')


def build_app(detector=None):
    detector = detector or TrafficSignDetector()

    def respond(image, question, threshold, mode, state):
        try:
            response = ask(detector, image, question, threshold, mode, state)
            return (response['state']['history'], response['image'], response['result'],
                    response['trace'], context_view(response['state']), response['state'])
        except Exception as exc:
            # Do not expose provider headers, API keys or filesystem tracebacks in UI.
            if isinstance(exc, ValueError):
                raise gr.Error(str(exc)) from None
            print(f'调用失败：{type(exc).__name__}', flush=True)
            raise gr.Error('调用失败，请检查模型/API 配置；本轮未生成回答。') from None

    def reset():
        return [], None, {}, [], [], new_state()

    def parse_directions(image, threshold, direction_state):
        revision = direction_state['revision']
        try:
            report = read_directions(image, detector, threshold)
            path = OUTPUTS / 'direction_ui' / f'{uuid.uuid4().hex}.jpg'
            draw_evidence(image, report, path)
            if revision != direction_state['revision']:
                return gr.skip(), gr.skip(), gr.skip()
            return summarize_directions(report), str(path), report
        except ValueError as exc:
            raise gr.Error(str(exc)) from None
        except Exception as exc:
            print(f'方向解析失败：{type(exc).__name__}', flush=True)
            raise gr.Error('方向解析失败，请检查本地检测和 OCR 环境。') from None

    def invalidate_directions(direction_state):
        # State is session-local. In-place revision change also reaches any
        # running callback, which must discard its now-stale result.
        direction_state['revision'] += 1
        return '', None, {}, direction_state

    with gr.Blocks(title='Traffic Vision Agent', theme=gr.themes.Soft(primary_hue='teal')) as demo:
        gr.Markdown('# Traffic Vision Agent\n### 交通场景视觉查询 · 夜间模型演示\n选择图片，查询类别、数量与置信度，再生成检测框。')
        state = gr.State(new_state())
        direction_state = gr.State({'revision': 0})
        with gr.Row():
            with gr.Column(scale=5):
                image = gr.Image(type='pil', label='道路图片', sources=['upload'], height=350)
                gr.Examples([[str(p)] for p in sorted((ROOT/'test_images').glob('*.jpg'))], inputs=image)
                threshold = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label='检测置信度阈值')
                mode = gr.Radio(['离线规则', 'DeepSeek Tool Calling'], value='DeepSeek Tool Calling', label='问答模式')
                gr.Markdown('模型类别：`wran` · `ban` · `point-l` · `point-s`。类别语义待核对，暂保留训练标签。\n\n'
                            'DeepSeek 模式会发送问题、折叠后的文字上下文和结构化检测结果到配置的服务；图片保留在本机。')
                with gr.Accordion('标志语义知识库状态', open=False):
                    gr.Markdown(knowledge_status())
            with gr.Column(scale=7):
                chat = gr.Chatbot(type='messages', label='多轮问答', height=350)
                question = gr.Textbox(label='你的问题', placeholder='有多少个 ban？把它们框出来。')
                with gr.Row():
                    submit = gr.Button('分析 / 发送', variant='primary')
                    clear = gr.Button('清空会话')
                gr.Examples([['图里有什么交通标志？'], ['有多少个 ban？'], ['哪个置信度最高？'],
                             ['把它们框出来。'], ['ban 是什么意思？'], ['这块牌想表达什么？']],
                            inputs=question)
        plotted = gr.Image(label='检测结果', height=440)
        with gr.Accordion('结构化结果与工具调用记录', open=False):
            result = gr.JSON(label='检测 JSON')
            trace = gr.JSON(label='工具轨迹（cached 表示复用本会话检测结果）')
            context_panel = gr.Dataframe(headers=['上下文键', '值'], label='折叠后的会话上下文（Context Compression）',
                                         wrap=True, interactive=False)
        inputs = [image, question, threshold, mode, state]
        outputs = [chat, plotted, result, trace, context_panel, state]
        submit.click(respond, inputs, outputs, concurrency_limit=1, api_name='ask')
        question.submit(respond, inputs, outputs, concurrency_limit=1)
        for component in (image, threshold, mode):
            component.change(reset, outputs=outputs, queue=True)
        clear.click(reset, outputs=outputs)
        chat.clear(reset, outputs=outputs)
        with gr.Accordion('简单指路牌方向解析 · 试验版', open=True):
            gr.Markdown('使用上方道路原图，点击后自动检测牌面并读取文字和独立箭头。'
                        '支持独立分行箭头，以及简单的一箭头对应多个地名；方向限于直行、左转、右转。'
                        '四角清晰时会尝试校正倾斜牌面；文字框覆盖箭头时会重新核对文字。'
                        '复杂路网、模糊或关联不明确的部分会说明原因。试验版尚未通过冻结测试验收。'
                        '此功能完全在本机运行。黄色框为牌面、橙色为校正边界、蓝色为文字、紫色为箭头、绿色连线为已输出关系。')
            direction_examples=[ROOT.parent/'Private-dataset-master'/'val'/'images'/'08839.jpg',
                                ROOT.parent/'Private-dataset-master'/'train'/'images'/'08785.jpg']
            existing_examples=[[str(p)] for p in direction_examples if p.is_file()]
            if existing_examples:
                gr.Examples(existing_examples,inputs=image,label='开发示例：共享箭头 / 独立分行箭头')
            direction_button = gr.Button('解析方向并显示证据')
            direction_summary = gr.Markdown()
            direction_image = gr.Image(label='方向证据（原图坐标）', height=500)
            direction_json = gr.JSON(label='文字框、箭头框、关联与不确定原因', open=False)
            direction_button.click(parse_directions, [image, threshold, direction_state],
                                   [direction_summary, direction_image, direction_json],
                                   concurrency_limit=1, api_name='read_directions')
            for component in (image, threshold):
                component.change(invalidate_directions, [direction_state],
                                 outputs=[direction_summary, direction_image, direction_json, direction_state],
                                 queue=False)
        gr.Markdown('离线模式为规则路由基线；DeepSeek 模式提供 LLM 工具调用。'
                    '检测结果不能用于行车安全决策；标志含义只有在知识库收录依据后才会回答。')
    return demo


if __name__ == '__main__':
    build_app().queue(max_size=16).launch(server_name='127.0.0.1', server_port=7860,
                                         share=False, allowed_paths=[str(OUTPUTS)])
