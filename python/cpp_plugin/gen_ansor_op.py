import re

import numpy as np
import tvm
from tvm import te, tir, auto_scheduler, topi
import os
import json
import logging

USE_CACHE = True
logger = logging.getLogger('InfiniTensor')
logger.setLevel(logging.DEBUG)


def gen_ansor_op(input_tensors, input_dtypes, output_tensor, output_dtype, f,
                 func_name, input_names, output_name, nnet_expression: str,
                 nnet_simplified_expression: str, hash_code=None):
    assert len(input_tensors) == len(input_dtypes)
    assert len(input_tensors) == len(input_names)

    logging.debug(f'Work on hash {hash_code}')

    dir_name = os.path.join(".cache", "generated_kernels", str(hash_code))
    func_code_fn = os.path.join(dir_name, "kernel.cu")
    invoke_code_fn = os.path.join(dir_name, "invoke.cpp")
    config_fn = os.path.join(dir_name, "config.json")

    if USE_CACHE and hash_code is not None:
        if os.path.exists(dir_name):
            print(f"Use cache in {dir_name}")
            with open(func_code_fn, "r") as func_code_fin:
                func_code = func_code_fin.read()
            with open(invoke_code_fn, "r") as invoke_code_fin:
                invoke_code = invoke_code_fin.read()
            with open(config_fn, "r") as config_fin:
                config = json.loads(config_fin.read().strip())
                conv_time = config["conv_time"]
                invoke_params = config["invoke_params"]

            logger.debug(f'Find tuning log for {hash_code}')
            return func_code, invoke_code, conv_time, invoke_params

    print("Generating Ansor op: ")
    print(f)

    @auto_scheduler.register_workload(func_name)
    def compute():
        _locals = locals()
        exec(f, {'tvm': tvm, 'te': te, 'tir': tir, 'topi': topi}, _locals)
        return _locals['ret']

    target = tvm.target.Target("cuda")

    task = auto_scheduler.SearchTask(func=func_name, args=(), target=target)

    # Inspect the computational graph
    print("Computational DAG:")
    print(task.compute_dag)

    log_file = f"ansor_{func_name}_log.json"
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=10,
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

    def test_mutator():
        # test part
        tgt_temp = tvm.target.Target(target="llvm", host="llvm")
        all_tensors = compute()
        sch = te.create_schedule(all_tensors[0].op)
        args = all_tensors
        C0, K0, A0 = args
        func_temp = tvm.build(sch, args, tgt_temp, name="temp")

        # print result
        n, c, h, w, f, r, s = 1, 1, 2, 2, 1, 4, 4
        dev_temp = tvm.device(tgt_temp.kind.name, 0)
        A_temp = tvm.nd.array(
            np.arange(n*h*w*f).reshape(n, h, w, f).astype(A0.dtype), dev_temp)
        K_temp = tvm.nd.array(
            np.arange(f*r*s*c).reshape(f, r, s, c).astype(K0.dtype), dev_temp)
        C_temp = tvm.nd.array(
            np.zeros((1, 4, 4, 1)).astype(C0.dtype), dev_temp)
        func_temp(C_temp, K_temp, A_temp)
        print("================= Test Result =====================")
        print(C_temp)

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
            map(str, thread_dim)), ", ".join(input_names), output_name)
    invoke_params = block_dim + thread_dim

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
    func_code += "\n/* NNET tensor expression \n" + nnet_expression + "\n*/\n"
    func_code += "\n/* NNET simplified tensor expression \n" + \
        nnet_simplified_expression + "\n*/\n"
    func_code += "\n/* TVM compute\n" + f + "\n*/\n"
    print(func_code)
    print("Invoke Code")
    print(invoke_code)
    print("Time")
    print(conv_time)

    if hash_code is not None:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        with open(func_code_fn, "w") as func_code_fout:
            func_code_fout.write(func_code)
        with open(invoke_code_fn, "w") as invoke_code_fout:
            invoke_code_fout.write(invoke_code)
        with open(config_fn, "w") as config_fout:
            config_fout.write(json.dumps({
                "conv_time": conv_time,
                "invoke_params": invoke_params
            }, ensure_ascii=False, indent=2))

    return func_code, invoke_code, conv_time, invoke_params  # ms
