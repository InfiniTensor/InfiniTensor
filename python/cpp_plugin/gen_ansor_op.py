import re

import numpy as np
import tvm
from tvm import te, tir, auto_scheduler, topi


def gen_ansor_op(input_tensors, input_dtypes, output_tensor, output_dtype, f, func_name, input_names, output_name):
    assert len(input_tensors) == len(input_dtypes)
    assert len(input_tensors) == len(input_names)

    print("Generating Ansor op: ")
    print(f)

    @auto_scheduler.register_workload(func_name)
    def compute():
        _locals = locals()
        exec(f, {'tvm': tvm, 'te': te, 'tir': tir}, _locals)
        return _locals['ret']

    target = tvm.target.Target("cuda")

    task = auto_scheduler.SearchTask(func=func_name, args=(), target=target)

    # Inspect the computational graph
    print("Computational DAG:")
    print(task.compute_dag)

    log_file = f"ansor_{func_name}_log.json"
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=500,
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2,
    )

    # Run auto-tuning (search)
    task.tune(tune_option)
    # Apply the best schedule
    sch, args = task.apply_best(log_file)

    # Kill the measurement process
    del measure_ctx

    ir = str(tvm.lower(sch, args, simple_mode=True))
    thread_dim = [1, 1, 1]
    block_dim = [1, 1, 1]
    p = re.compile('"thread_extent" = (\d+)')
    for line in ir.splitlines():
        if "thread_extent" in line:
            ext = int(p.search(line).group(1))
            if "threadIdx.x" in line:
                thread_dim[0] = ext
            elif "threadIdx.y" in line:
                thread_dim[1] = ext
            elif "threadIdx.z" in line:
                thread_dim[2] = ext
            elif "blockIdx.x" in line:
                block_dim[0] = ext
            elif "blockIdx.y" in line:
                block_dim[1] = ext
            elif "blockIdx.z" in line:
                block_dim[2] = ext

    func = tvm.build(sch, args, target, name=func_name)

    func_code = func.imported_modules[0].get_source()
    invoke_code = "%s_kernel0<<<dim3(%s), dim3(%s)>>>(%s, %s);" % (
        func_name, ", ".join(map(str, block_dim)), ", ".join(
            map(str, thread_dim)),
        output_name, ", ".join(input_names))
    invoke_params = thread_dim + block_dim
    
    ctx = tvm.cuda(0)
    input_a = []
    for i, (shape, dtype) in enumerate(zip(input_tensors, input_dtypes)):
        a_np = np.random.uniform(size=shape).astype(dtype)
        input_a.append(tvm.nd.array(a_np, ctx))
    a_out = tvm.nd.array(np.zeros(output_tensor, dtype=output_dtype), ctx)
    func(a_out, *input_a)
    evaluator = func.time_evaluator(func.entry_name, ctx, number=100)
    conv_time = evaluator(a_out, *input_a).mean * 1e3

    print("Func Code")
    # Attach TVM code behind func_code
    func_code += "\n/* " + f + "*/"
    print(func_code)
    print("Invoke Code")
    print(invoke_code)
    print("Time")
    print(conv_time)

    return func_code, invoke_code, conv_time, invoke_params # ms
