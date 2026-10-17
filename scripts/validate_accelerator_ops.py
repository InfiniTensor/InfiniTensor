"""Execute InfiniTensor graphs on the requested accelerator; never fall back."""
import argparse
import json
import os
import platform
import time
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as H, numpy_helper as N, TensorProto as T
from pyinfinitensor import backend
from pyinfinitensor.onnx import OnnxStub
from pyinfinitensor.embedding import fp32_inference_policy

p = argparse.ArgumentParser()
p.add_argument('--device', required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
report = {'device': a.device, 'host': platform.node(), 'cases': [], 'fallback': False}
runtime = backend.runtime(a.device, 0)
report['runtime_class'] = type(runtime).__name__
rng = np.random.default_rng(315)

def model(nodes, inputs, outputs, weights=()):
    result = H.make_model(H.make_graph(nodes, 'accelerator-validation', inputs, outputs, list(weights)), opset_imports=[H.make_opsetid('', 17)])
    result.ir_version = 8
    return result

def check(name, fn):
    started = time.perf_counter()
    try:
        detail = fn()
        item = {'name': name, 'status': 'PASS', **(detail or {})}
    except Exception as e:
        item = {'name': name, 'status': 'FAIL', 'error': str(e)}
    item['seconds'] = time.perf_counter()-started
    report['cases'].append(item)
    print(json.dumps(item), flush=True)

def dynamic():
    vi = lambda name: H.make_tensor_value_info(name, T.FLOAT, ['B', 'S'])
    m = model([H.make_node('Add', ['x','w'], ['z']), H.make_node('Mul',['z','w'],['y'])], [vi('x')], [vi('y')], [N.from_array(np.asarray([2.0],np.float32),'w')])
    stub = OnnxStub(m, runtime)
    previous = None
    shapes = [(1,17),(7,83),(3,511),(16,128),(1,17)]
    shapes += [(int(rng.integers(1,33)), int(rng.integers(1,513))) for _ in range(100)]
    for b,s in shapes:
        x = rng.standard_normal((b,s)).astype(np.float32)
        y = stub.run_numpy({'x':x})['y']
        np.testing.assert_allclose(y,(x+2)*2,atol=1e-5,rtol=1e-5)
        if previous is not None:
            np.testing.assert_array_equal(previous[0],previous[1])
        previous = (y,y.copy())
    assert stub.weight_upload_count == 1
    return {'requests':len(shapes), 'weight_upload_count':stub.weight_upload_count}

def matmul():
    x=rng.standard_normal((2,4,8)).astype(np.float32)
    w=rng.standard_normal((8,6)).astype(np.float32)
    m=model([H.make_node('MatMul',['x','w'],['y'])], [H.make_tensor_value_info('x',T.FLOAT,[2,4,8])],[H.make_tensor_value_info('y',T.FLOAT,[2,4,6])],[N.from_array(w,'w')])
    y=OnnxStub(m,runtime).run_numpy({'x':x})['y']
    np.testing.assert_allclose(y,x@w,rtol=1e-4,atol=1e-4)

def shape_parameters():
    nodes = [H.make_node('Shape',['x'],['shape']),
             H.make_node('Gather',['shape','zero'],['b']),
             H.make_node('Unsqueeze',['b','axes'],['bv']),
             H.make_node('Concat',['bv','minus_one'],['target'],axis=0),
             H.make_node('Reshape',['x','target'],['y'])]
    m=model(nodes,[H.make_tensor_value_info('x',T.FLOAT,['B','S',2])],
            [H.make_tensor_value_info('y',T.FLOAT,['B','T'])],
            [N.from_array(np.array(0,np.int64),'zero'),N.from_array(np.array([0],np.int64),'axes'),N.from_array(np.array([-1],np.int64),'minus_one')])
    stub=OnnxStub(m,runtime)
    for b,s in [(1,17),(7,83),(3,511),(16,128),(1,17)]:
        x=rng.standard_normal((b,s,2)).astype(np.float32)
        y=stub.run_numpy({'x':x})['y']
        np.testing.assert_array_equal(y,x.reshape(b,s*2))
    return {'requests':5,'weight_upload_count':stub.weight_upload_count}

def softmax():
    x=rng.standard_normal((2,4,17)).astype(np.float32)
    m=model([H.make_node('Softmax',['x'],['y'],axis=-1)],[H.make_tensor_value_info('x',T.FLOAT,list(x.shape))],[H.make_tensor_value_info('y',T.FLOAT,list(x.shape))])
    y=OnnxStub(m,runtime).run_numpy({'x':x})['y']
    expected=np.exp(x-x.max(-1,keepdims=True)); expected/=expected.sum(-1,keepdims=True)
    np.testing.assert_allclose(y,expected,rtol=1e-4,atol=1e-5)

def gather():
    w=np.arange(32,dtype=np.float32).reshape(8,4)
    m=model([H.make_node('Gather',['w','x'],['y'],axis=0)],[H.make_tensor_value_info('x',T.INT64,[2])],[H.make_tensor_value_info('y',T.FLOAT,[2,4])],[N.from_array(w,'w')])
    np.testing.assert_array_equal(OnnxStub(m,runtime).run_numpy({'x':np.array([1,3],np.int64)})['y'],w[[1,3]])

def layernorm():
    x=rng.standard_normal((2,4,8)).astype(np.float32)
    m=model([H.make_node('LayerNormalization',['x','w','b'],['y'],axis=-1,epsilon=1e-5)],[H.make_tensor_value_info('x',T.FLOAT,[2,4,8])],[H.make_tensor_value_info('y',T.FLOAT,[2,4,8])],[N.from_array(np.ones(8,np.float32),'w'),N.from_array(np.zeros(8,np.float32),'b')])
    y=OnnxStub(m,runtime).run_numpy({'x':x})['y']
    np.testing.assert_allclose(y,(x-x.mean(-1,keepdims=True))/np.sqrt(x.var(-1,keepdims=True)+1e-5),atol=1e-4,rtol=1e-4)

def gather_nd_indices():
    data = rng.standard_normal((2,5,3)).astype(np.float32)
    for indices in [np.array(-1,np.int64), np.array([[0,-1],[2,-3]],np.int64)]:
        expected = np.take(data, indices, axis=1)
        m=model([H.make_node('Gather',['data','indices'],['y'],axis=-2)],
                [H.make_tensor_value_info('data',T.FLOAT,list(data.shape)),
                 H.make_tensor_value_info('indices',T.INT64,list(indices.shape))],
                [H.make_tensor_value_info('y',T.FLOAT,list(expected.shape))])
        y=OnnxStub(m,runtime).run_numpy({'data':data,'indices':indices})['y']
        np.testing.assert_array_equal(y,expected)

def encoder_math():
    import math
    x=rng.standard_normal((3,7,5)).astype(np.float32)
    w=np.array([2.0],np.float32)
    for op, expected in [('Sub',x-w),('Div',x/w),('Pow',x**w)]:
        m=model([H.make_node(op,['x','w'],['y'])],
            [H.make_tensor_value_info('x',T.FLOAT,list(x.shape))],
            [H.make_tensor_value_info('y',T.FLOAT,list(x.shape))],[N.from_array(w,'w')])
        np.testing.assert_allclose(OnnxStub(m,runtime).run_numpy({'x':x})['y'],expected,atol=1e-5,rtol=1e-4)
    for op, data, expected in [('Erf',x,np.vectorize(math.erf)(x)),('Sqrt',x*x,np.abs(x))]:
        m=model([H.make_node(op,['x'],['y'])],
            [H.make_tensor_value_info('x',T.FLOAT,list(x.shape))],
            [H.make_tensor_value_info('y',T.FLOAT,list(x.shape))])
        np.testing.assert_allclose(OnnxStub(m,runtime).run_numpy({'x':data})['y'],expected,atol=1e-5,rtol=1e-4)

def cast_int64_float():
    x=np.array([[0,1,-4],[512,10000,-32767]],np.int64)
    m=model([H.make_node('Cast',['x'],['y'],to=T.FLOAT)],
        [H.make_tensor_value_info('x',T.INT64,list(x.shape))],
        [H.make_tensor_value_info('y',T.FLOAT,list(x.shape))])
    np.testing.assert_array_equal(OnnxStub(m,runtime).run_numpy({'x':x})['y'],x.astype(np.float32))

def layernorm_axes():
    x=rng.standard_normal((2,3,5)).astype(np.float32)
    w=rng.standard_normal((3,5)).astype(np.float32)
    bias=rng.standard_normal((3,5)).astype(np.float32)
    m=model([H.make_node('LayerNormalization',['x','w','bias'],['y'],axis=1,epsilon=0.01)],
        [H.make_tensor_value_info('x',T.FLOAT,list(x.shape))],
        [H.make_tensor_value_info('y',T.FLOAT,list(x.shape))],
        [N.from_array(w,'w'),N.from_array(bias,'bias')])
    expected=(x-x.mean((1,2),keepdims=True))/np.sqrt(x.var((1,2),keepdims=True)+0.01)*w+bias
    np.testing.assert_allclose(OnnxStub(m,runtime).run_numpy({'x':x})['y'],expected,atol=1e-5,rtol=1e-4)

def attention_matmul_broadcast():
    x=rng.standard_normal((2,1,4,8)).astype(np.float32)
    w=rng.standard_normal((1,3,8,6)).astype(np.float32)
    m=model([H.make_node('MatMul',['x','w'],['y'])],
        [H.make_tensor_value_info('x',T.FLOAT,list(x.shape)),
         H.make_tensor_value_info('w',T.FLOAT,list(w.shape))],
        [H.make_tensor_value_info('y',T.FLOAT,[2,3,4,6])])
    np.testing.assert_allclose(OnnxStub(m,runtime).run_numpy({'x':x,'w':w})['y'],x@w,atol=1e-5,rtol=1e-4)

with fp32_inference_policy():
    for name,fn in [('dynamic_add_mul',dynamic),('dynamic_shape_parameters',shape_parameters),('matmul',matmul),('softmax',softmax),('gather',gather),('layernorm',layernorm)]:
        check(name,fn)
    for fn in (gather_nd_indices, encoder_math, cast_int64_float, layernorm_axes, attention_matmul_broadcast):
        check(fn.__name__, fn)
report['status'] = 'FAIL' if any(item['status']=='FAIL' for item in report['cases']) else 'PASS'
Path(a.output).write_text(json.dumps(report,indent=2),encoding='utf-8')
raise SystemExit(int(any(item['status']=='FAIL' for item in report['cases'])))
