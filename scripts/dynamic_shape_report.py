"""Generate the first-person project report from recorded validation evidence."""
import argparse
import json
import re
from html import escape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", default="output/validation")
    parser.add_argument("--output", required=True)
    parser.add_argument("--author", default="uel0p")
    parser.add_argument("--pr-url", default="")
    parser.add_argument("--font", default="")
    args = parser.parse_args()
    evidence = Path(args.evidence)
    synthetic = json.loads((evidence / "results.json").read_text(encoding="utf-8"))
    real = json.loads((evidence / "real-model/results.json").read_text(encoding="utf-8"))
    cuda = json.loads((evidence / "cuda/results.json").read_text(encoding="utf-8"))
    cpu_tests = (evidence / "cpp-tests.log").read_text(encoding="utf-8", errors="replace")
    cuda_tests = (evidence / "cuda-tests.log").read_text(encoding="utf-8", errors="replace")
    python_tests = (evidence / "python-cuda-tests.log").read_text(encoding="utf-8", errors="replace")
    if "100% tests passed, 0 tests failed out of 41" not in cpu_tests:
        raise ValueError("CPU CTest evidence is missing")
    if "100% tests passed, 0 tests failed out of 78" not in cuda_tests:
        raise ValueError("CUDA CTest evidence is missing")
    if not re.search(r"Ran 95 tests.*\n\nOK", python_tests, re.S):
        raise ValueError("CUDA Python regression evidence is missing")
    if len(real["runs"]) != 5 or not real["same_model_instance"]:
        raise ValueError("Real-model evidence is incomplete")
    graph = cuda["cuda_graph"]
    if graph["capture_count_after_first_cycle"] != graph["capture_count_after_second_cycle"]:
        raise ValueError("CUDA Graph cache was not reused")

    fonts = [args.font, "C:/Windows/Fonts/msyh.ttc", "/mnt/c/Windows/Fonts/msyh.ttc"]
    font = next((item for item in fonts if item and Path(item).is_file()), None)
    if not font:
        raise ValueError("A Chinese TrueType font is required")
    pdfmetrics.registerFont(TTFont("Chinese", font, subfontIndex=0))
    navy, teal = colors.HexColor("#17324D"), colors.HexColor("#167D8D")
    pale, line, gray = colors.HexColor("#EAF3F5"), colors.HexColor("#D3DEE6"), colors.HexColor("#53677A")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("BodyCN", fontName="Chinese", fontSize=9.7, leading=16.5, textColor=navy, spaceAfter=7, wordWrap="CJK"))
    styles.add(ParagraphStyle("SmallCN", parent=styles["BodyCN"], fontSize=8, leading=12, textColor=gray, spaceAfter=5))
    styles.add(ParagraphStyle("TitleCN", parent=styles["BodyCN"], fontSize=24, leading=34, spaceAfter=14))
    styles.add(ParagraphStyle("SubTitleCN", parent=styles["BodyCN"], fontSize=15, leading=23, textColor=teal, spaceAfter=14))
    styles.add(ParagraphStyle("HeadingCN", parent=styles["BodyCN"], fontSize=16, leading=24, spaceAfter=12))
    styles.add(ParagraphStyle("CellCN", parent=styles["BodyCN"], fontSize=8.2, leading=13, spaceAfter=0))
    styles.add(ParagraphStyle("MetricCN", parent=styles["BodyCN"], fontSize=13, leading=18, alignment=TA_CENTER, textColor=teal, spaceAfter=2))
    styles.add(ParagraphStyle("CodeCN", fontName="Courier", fontSize=7.7, leading=11.5, textColor=navy, backColor=pale, borderPadding=7, spaceAfter=10))
    story = []

    def p(text, style="BodyCN"):
        story.append(Paragraph(text, styles[style]))

    def heading(number, title):
        p(f"{number:02d} / {title}", "HeadingCN")

    def page():
        story.append(PageBreak())

    def code(text):
        story.append(Preformatted(text, styles["CodeCN"]))

    def table(rows, widths, header=True):
        data = [[Paragraph(escape(str(value)), styles["CellCN"]) for value in row] for row in rows]
        item = Table(data, colWidths=[value * mm for value in widths], repeatRows=1 if header else 0)
        commands = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6), ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5), ("LINEBELOW", (0, 0), (-1, -1), 0.25, line)]
        if header:
            commands += [("BACKGROUND", (0, 0), (-1, 0), pale), ("LINEBELOW", (0, 0), (-1, 0), 1, teal)]
        item.setStyle(TableStyle(commands))
        story.extend([item, Spacer(1, 9)])

    real_max = max(row["max_abs_error"] for row in real["runs"])
    ordinary_max = max(row["max_abs_error"] for row in cuda["ordinary_cuda"]["runs"])
    memory = synthetic["memory_benchmark"]
    exact, reuse = memory["exact_fit"], memory["high_water_reuse"]
    latency_drop = (exact["median_end_to_end_us"] - reuse["median_end_to_end_us"]) / exact["median_end_to_end_us"] * 100
    invalidation = graph["storage_invalidation"]

    p("2026 夏季训练营 · AI 编译器方向", "SmallCN")
    story.append(Spacer(1, 20 * mm))
    p("ONNX 动态 Shape<br/>子图编译与执行支持", "TitleCN")
    p("设计、实现、验证与复现报告", "SubTitleCN")
    table([["项目提交人", args.author], ["项目仓库", "https://github.com/InfiniTensor/InfiniTensor"],
           ["开发基线 / 分支", "5cb7c1f / dynamic-shape"],
           ["GitHub PR", args.pr_url or "尚未创建；我会在手动创建 PR 后补填"],
           ["验证日期", "2026-09-20"]], [42, 124], header=False)
    p("我完成了基础动态 Shape 链路，并继续验证了优秀标准 1-5。我的实现覆盖动态维度保存、Shape Tensor 求值、双输入 Reshape、动态 H/W 传播、CPU/CUDA 执行、CUDA Graph 缓存与失效，以及高水位内存容量复用。")
    table([["验收项", "本机实测结果"], ["基础链路", "Shape → Gather → Unsqueeze → Concat → Reshape"],
           ["真实模型", f"SqueezeNet 1.0，5 组 H/W，最大绝对误差 {real_max:.3g}"],
           ["CUDA / CUDA Graph", f"普通 CUDA 误差 {ordinary_max:g}；4 个 Shape 捕获 4 次，第二轮仍为 4 次"],
           ["完整回归", "CPU CTest 41/41；CUDA CTest 78/78；Python 95/95"]], [45, 121])

    page(); heading(1, "我对问题的理解")
    p("一开始我把问题分成两类：一类是 Tensor 的逻辑 Shape，另一类是计算图里真正承载维度值的 Shape Tensor。只修改输入 Tensor 的 dims，只能解决前一类问题；当 Reshape 的第二个输入来自 Shape、Gather、Unsqueeze 和 Concat 时，导入阶段并不知道最终值。")
    p("因此我没有把动态维度简单替换成 1 后继续静态编译。我保存了 ONNX 输入声明，把小型整数 Shape 子图提取成主机执行计划，并在每次内存规划之前求值。数据图仍由 InfiniTensor 原有 CPU/CUDA kernel 执行。")
    code("Load ONNX once\n  -> keep fixed / symbolic / unknown dimension schema\n  -> compile and partially evaluate Shape program\n\nset_input(actual shape)\n  -> validate schema\n  -> evaluate Shape tensors\n  -> update input and target tensors\n  -> Graph::shape_infer()\n  -> Graph::dataMalloc()\n  -> CPU / CUDA / CUDA Graph execution")
    table([["层次", "我做的改动"], ["ONNX 前端", "保存维度声明；提取 Shape 依赖；每轮求值并物化 int64 Tensor"],
           ["Operator", "为 Reshape 增加真正的第二 Tensor 输入；保留静态列表接口"],
           ["Graph", "在 Shape 更新后复用现有 shape_infer 和动态内存规划"],
           ["真实 H/W", "修复 Pooling 缓存旧 N/C/H/W，保证 Shape 能继续向后传播"],
           ["观测", "暴露逻辑计划字节数、池容量、存储 ID 和分配代次"]], [37, 129])
    p("我始终复用同一个 OnnxStub 和同一个 C++ Graph，没有在 Shape 变化时重新加载模型。ONNX Runtime 只作为独立正确性参照，没有参与 InfiniTensor 的实际计算。", "SmallCN")

    page(); heading(2, "动态维度与 Shape Tensor")
    table([["ONNX 声明", "内部保存", "运行时规则"], ["dim_value=3", "整数 3", "实际值必须仍为 3"],
           ["dim_param=batch", "字符串 batch", "允许变化；同名符号本轮必须一致"],
           ["未设置", "None", "允许变化；不同匿名维度不绑定"], ["实际输入", "正整数列表", "Rank 不变，且不超过 int32"]], [39, 37, 90])
    p("我用 protobuf 的 WhichOneof('value') 区分固定值、符号和匿名未知，没有引入复杂的符号表达式系统。C++ Tensor 始终只保存本轮的具体 Shape。对于 7×7 卷积这类不能用 1×1 初始化的模型，我通过 input_shapes 传入第一个合法尺寸。")
    table([["算子", "本项目支持的范围"], ["Shape", "读取图输入或 initializer 的元数据；支持 start/end"],
           ["Gather", "int32/int64 索引、负索引和越界检查"], ["Unsqueeze / Squeeze", "静态 axes；标量与向量转换；非法轴检查"],
           ["Concat", "按 axis 拼接并检查 dtype 一致"], ["Cast", "Shape 链中的 int32/int64 转换"],
           ["动态 Reshape", "rank-1 int64 目标；支持 0 和单个 -1；检查元素数"]], [42, 124])
    p("Shape 计划只处理元数据依赖。我没有把这些 NumPy 求值器包装成通用数据 kernel；它们的用途是尽早得到后续 Tensor 的具体 Shape，再交给原有 Runtime。")

    page(); heading(3, "每轮执行、异常恢复和兼容性")
    p("我把 Shape Tensor 标记为长生命周期存储，避免激活值复用覆盖它；每次 set_input 先计算新 Shape 值，再调用图级推导和内存规划，最后恢复 initializer 和 Shape 值。输入数据要在规划完成后再 copyin，避免扩大 Shape 时写入容量不足的旧地址。")
    table([["风险", "我的处理与验证"], ["输入由小变大", "重新规划；容量不足时更换底层存储；CPU/CUDA 均覆盖"],
           ["输入由大变小", "按当前逻辑 Shape 写入和读取；与 ORT 连续对齐"],
           ["权重被破坏", "重分配后恢复 initializer；保留原有 MatMul 回归"],
           ["非法 Reshape", "恢复旧输入和 Shape Tensor 后重新规划，模型仍可继续运行"],
           ["静态模型", "initializer Reshape 继续走原接口；全量旧测试保持通过"],
           ["导出", "保留源 ModelProto 的 dim_param 和 Shape 子图，不固化当前尺寸"]], [46, 120])
    p("真实视觉网络还暴露出 PoolingObj 的旧问题：它在构造时缓存 h/w，后续推导仍使用旧值。我把 n/c/h/w 的刷新放进 inferShape()，这样输出 Shape 和 CUDA kernel 查询到的工作负载参数会同时更新，并增加了 C++ 回归测试。")

    page(); heading(4, "优秀标准 1：Shape 子图部分求值")
    p("我用 None 表示编译期未知维度。例如输入 [batch, 2, 3] 的抽象 Shape 是 [None, 2, 3]。batch 分支必须留到运行时，但 Gather(index=1) 仍能得到常量 2，后续 Unsqueeze、Cast、Squeeze 可以继续折叠。")
    code("Shape(x) = [None, 2, 3]\n  +-> Gather(0) -> Unsqueeze --------+\n  +-> Gather(1) -> Unsqueeze -> Cast32 -> Cast64\n      -> Squeeze -> Unsqueeze [compile-time: 2]\n                                      +-> Concat -> Reshape")
    table([["项目", "关闭优化", "开启优化"], ["Shape 节点总数", "10", "10"], ["编译期折叠", "0", "6"],
           ["每轮运行时节点", "10", "4"],
           ["batch 模型求值中位数", f"{synthetic['cases'][0]['shape_eval_median_us']:.3f} μs", f"{synthetic['cases'][1]['shape_eval_median_us']:.3f} μs"],
           ["H/W 模型求值中位数", f"{synthetic['cases'][2]['shape_eval_median_us']:.3f} μs", f"{synthetic['cases'][3]['shape_eval_median_us']:.3f} μs"]], [65, 50, 50])
    p("节点数从 10 降到 4 是确定性的 60% 运行时计算量减少。微秒数据只测 Shape 计划，不包含数据 kernel 和内存规划；我不把它解释成整个模型固定加速多少倍。", "SmallCN")

    page(); heading(5, "优秀标准 2：真实动态 H/W 网络")
    p("我按公开的 SqueezeNet 1.0 feature extractor 拓扑构造 ONNX：首层 7×7 Conv、三次 MaxPool、8 个 Fire 模块，最终得到 512 通道特征。为保证模型可完全离线复现，我使用固定随机种子生成权重；它不是预训练分类模型，也不用于评价准确率。")
    p("在网络末尾，我用输入的 batch 构造 [batch, 512, -1]，再动态 Reshape 特征图。下面五轮只创建一次模型实例。")
    rows = [["轮次", "输入 NCHW", "输出", "最大绝对误差", "池容量 / B"]]
    for index, row in enumerate(real["runs"], 1):
        rows.append([index, str(row["input"]), str(row["output"]), f"{row['max_abs_error']:.3g}", row["activation_pool_capacity"]])
    table(rows, [15, 51, 40, 32, 28])
    p(f"五轮 float32 结果都满足 rtol=2e-4、atol=2e-5，最大绝对误差为 {real_max:.3g}。36×36 回归时，逻辑需求降为 190,784 字节，但池仍保留 602,688 字节，存储 ID 也保持不变。")
    p("我选择的 H/W 使三次 ceil-mode MaxPool 窗口都整除。现有 MaxPool 在其他边界尺寸与 ORT 有取整差异，这个问题不属于本项目的动态 Shape 主线，我把它列为已知限制。", "SmallCN")

    page(); heading(6, "优秀标准 3-4：CUDA 与 CUDA Graph")
    p("我在 RTX 3060 Laptop GPU、CUDA Toolkit 12.9 上构建了独立的 CudaDebug 目录。普通 CUDA 使用同一实例运行 1→2→8→3→1，所有输出与 ORT 完全一致，最大绝对误差为 0。")
    table([["CUDA Graph 检查", "实测值"], ["第一轮不同 Shape", "1、2、8、3，共捕获 4 次，缓存 4 项"],
           ["第二轮相同序列", "捕获次数仍为 4，命中已有 Shape"], ["第一轮存储更换", "0 次，storage id 始终为 24"],
           ["扩容到 batch=32", f"storage {invalidation['storage_id_before_growth']} → {invalidation['grown_shape']['storage_id']}；捕获数 5；缓存清为 1"],
           ["回到 batch=1", f"捕获数 {invalidation['return_to_previous_shape']['capture_count']}；新地址下重新捕获，未误用旧图"]], [53, 113])
    p("CUDA Graph 的缓存、LRU 和地址失效框架来自项目基线。我新增的是动态 Shape 子图与双输入 Reshape 的接入、Python 端到端验证，以及扩容后旧图不得复用的证据。这样写能区分我完成的集成工作和仓库原有能力。")
    p("我还运行了仓库原生 test_cudagraph、test_cuda_reshape、test_cuda_pooling，最后执行完整 CUDA CTest，78/78 通过。", "SmallCN")

    page(); heading(7, "优秀标准 5：容量复用与 Benchmark")
    p("高水位容量复用算法已经存在于项目分配器中。我没有重复实现同一套算法，而是为它增加只读观测接口，并把它接入 ONNX 动态 Shape Demo。为了比较，我把每轮显式 trim 到精确容量作为 exact-fit 基线，与默认 high-water reuse 运行相同序列。")
    table([["200 轮 / 1000 次", "exact-fit", "high-water reuse"],
           ["底层存储更换次数", exact["storage_reallocations"], reuse["storage_reallocations"]],
           ["端到端中位延迟", f"{exact['median_end_to_end_us']:.3f} μs", f"{reuse['median_end_to_end_us']:.3f} μs"],
           ["峰值池容量", f"{exact['peak_capacity_bytes']} B", f"{reuse['peak_capacity_bytes']} B"]], [69, 48, 48])
    p(f"在这组小图上，高水位策略把观测到的存储更换从 {exact['storage_reallocations']} 次降到 0 次，中位端到端延迟下降约 {latency_drop:.1f}%。两种策略峰值容量相同；高水位策略的代价是输入缩小时不立即归还容量，可在需要时显式调用 trim_memory()。")
    p("计时范围是 set_input + 可选 trim + copyin + run + copyout。这个结果反映本机和这张小图，不应直接外推到大型模型；存储更换次数和容量则由 storage id/bytes 直接记录。", "SmallCN")

    page(); heading(8, "测试、环境与复现")
    env = synthetic["environment"]
    table([["项目", "环境 / 结果"], ["CPU", "Intel Core i7-13700H；WSL2 x86_64"],
           ["GPU", "NVIDIA GeForce RTX 3060 Laptop GPU，12 GB"], ["CUDA", "Toolkit 12.9；driver 610.74"],
           ["Python / NumPy", f"{env['python']} / {env['numpy']}"], ["ONNX / ORT", f"{env['onnx']} / {env['onnxruntime']}"],
           ["自动化", "CPU CTest 41/41；CUDA CTest 78/78；Python CUDA 构建 95/95"]], [49, 117])
    code("conda activate infinitensor\ncd /home/k/InfiniTensor\nmake build TYPE=Debug CUDA=OFF TEST=ON\nctest --test-dir build/Debug --output-on-failure -j4\npython -m unittest discover -s pyinfinitensor/tests -p 'test_*.py' -v\npython -m pyinfinitensor.dynamic_shape_demo --repeats 200 --output output/validation\npython -m pyinfinitensor.real_model_demo --output output/validation/real-model")
    code("cmake -S . -B build/CudaDebug -DCMAKE_BUILD_TYPE=Debug \\\n  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DUSE_CUDA=ON -DBUILD_TEST=ON\ncmake --build build/CudaDebug -j8\nctest --test-dir build/CudaDebug --output-on-failure -j4\ncp build/CudaDebug/backend*.so pyinfinitensor/src/pyinfinitensor/\npython -m pyinfinitensor.cuda_dynamic_shape_demo --output output/validation/cuda")
    p("我把人工模型、真实模型、CUDA 结果和测试日志都保存在 output/validation 下。报告脚本会先检查 41/41、78/78、95/95 以及 CUDA Graph 捕获计数，再允许生成 PDF。")

    page(); heading(9, "支持边界与总结")
    table([["范围", "当前结论"], ["已支持", "固定/符号/未知维度；Shape/Gather/Squeeze/Unsqueeze/Concat/Cast；动态 Reshape"],
           ["设备", "Native CPU、CUDA；CUDA Graph 按 Shape/地址安全缓存"], ["Shape 来源", "图输入或 initializer 的元数据"],
           ["尚未支持", "数据相关 Shape、动态 Rank、控制流、动态 Expand/ConstantOfShape、allowzero=1"],
           ["真实模型限制", "随机权重的 SqueezeNet 1.0 feature extractor，不代表预训练分类精度"],
           ["输入限制", "固定 Rank、正 int32 维度；暂不支持零长度输入"]], [47, 119])
    p("回顾整个实现，我认为最关键的不是单独增加几个 ONNX 算子，而是把运行时 Shape 值真正放到内存规划之前，并让同一张图在扩大、缩小、重复 Shape 和错误恢复后都能继续工作。真实网络和 CUDA 验证还帮我发现并修复了 Pooling 缓存旧尺寸的问题。")
    p("我最终完成了基础通过标准，并对优秀标准 1-5 都给出了代码、自动化测试和本机证据。PR 和邮件提交由我手动完成；创建 PR 后，我会用 --pr-url 重新生成首页。")
    p("参考资料", "SubTitleCN")
    for label, url in [("InfiniTensor", "https://github.com/InfiniTensor/InfiniTensor"),
                       ("ONNX IR", "https://onnx.ai/onnx/repo-docs/IR.html"),
                       ("ONNX Shape", "https://onnx.ai/onnx/operators/onnx__Shape.html"),
                       ("ONNX Reshape", "https://onnx.ai/onnx/operators/onnx__Reshape.html"),
                       ("SqueezeNet", "https://arxiv.org/abs/1602.07360")]:
        p(f'{label}：<link href="{url}" color="#167D8D">{url}</link>', "SmallCN")

    def frame(canvas, doc):
        canvas.saveState(); canvas.setStrokeColor(teal); canvas.line(22 * mm, 282 * mm, 188 * mm, 282 * mm)
        canvas.setFont("Chinese", 8); canvas.setFillColor(gray)
        canvas.drawString(22 * mm, 286 * mm, "InfiniTensor / ONNX Dynamic Shape")
        canvas.drawString(22 * mm, 13 * mm, f"{args.author} · 2026-09-20")
        canvas.drawRightString(188 * mm, 13 * mm, str(doc.page)); canvas.restoreState()

    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(str(output), pagesize=A4, leftMargin=22 * mm, rightMargin=22 * mm,
                                 topMargin=25 * mm, bottomMargin=23 * mm,
                                 title="ONNX 动态 Shape 子图编译与执行支持", author=args.author)
    document.build(story, onFirstPage=frame, onLaterPages=frame)
    print(output)


if __name__ == "__main__":
    main()
