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


def gen_ansor_so(input_tensors, input_dtypes, output_tensor, output_dtype,
                 tvm_code, func_name, nnet_expression: str,
                 nnet_simplified_expression: str, hash_code=None):
    assert len(input_tensors) == len(input_dtypes)
    
    logging.debug(f'Work on hash {hash_code}')
    dir_name = os.path.join(".cache", "generated_kernels", str(hash_code))
    
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    so_fn = os.path.join(dir_name, f"{func_name}.so")
    config_fn = os.path.join(dir_name, "config_so.json")
    
    print("Generating Ansor op: ")
    print(tvm_code)
    
    print("Input shape: ")
    print(input_tensors)
    print("Output shape: ")
    print(output_tensor)
    
    if USE_CACHE and hash_code is not None:
        if os.path.exists(dir_name) and \
            os.path.exists(so_fn) and \
            os.path.exists(config_fn):
            print(f"Use cache in {dir_name}")
            with open(config_fn, "r") as config_fin:
                config = json.loads(config_fin.read().strip())
                conv_time = config["conv_time"]

            logger.debug(f'Find tuning log for {hash_code}')
            return so_fn, conv_time
    
    @auto_scheduler.register_workload(func_name)
    def compute():
        _locals = locals()
        exec(tvm_code, {'tvm': tvm, 'te': te, 'tir': tir, 'topi': topi}, _locals)
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
    
    func = tvm.build(sch, args, target, name=func_name)
    func.export_library(so_fn)
    
    ctx = tvm.cuda(0)
    input_a = []
    for i, (shape, dtype) in enumerate(zip(input_tensors, input_dtypes)):
        a_np = np.random.uniform(size=shape).astype(dtype)
        input_a.append(tvm.nd.array(a_np, ctx))
    a_out = tvm.nd.array(np.zeros(output_tensor, dtype=output_dtype), ctx)
    func(a_out, *input_a)
    evaluator = func.time_evaluator(func.entry_name, ctx, number=100)
    conv_time = evaluator(a_out, *input_a).mean * 1e3
    
    print("====NNET tensor expression====")
    print(nnet_expression+"\n")
    print("====NNET simplified tensor expression====")
    print(nnet_simplified_expression+"\n")
    print("====Time====")
    print(conv_time)
    
    if USE_CACHE and hash_code is not None:
        with open(config_fn, "w") as config_fout:
            config_fout.write(json.dumps({
                "conv_time": conv_time,
            }, ensure_ascii=False, indent=2))
    
    return so_fn, conv_time
