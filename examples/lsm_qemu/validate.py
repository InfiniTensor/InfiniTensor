"""Differential acceptance tests for ONNX -> InfiniTensor -> QEMU -> RVNE."""
import argparse
import ctypes
import hashlib
import struct
import subprocess
from onnx import helper, numpy_helper, TensorProto
import json
import os
from pathlib import Path
import tempfile
import time
import numpy as np
import onnx
from pyinfinitensor import backend
from pyinfinitensor.onnx import OnnxStub
from export_model import sequence_model, reference_model, numpy_reference, DOMAIN

def audit_lines():
    path = os.environ.get('INFINI_RVNE_AUDIT')
    return [json.loads(line) for line in Path(path).read_text().splitlines()] if path and Path(path).exists() else []

def graph(model, naive=False):
    return OnnxStub(model, backend.cpu_runtime(), use_naive_allocator=naive, simplify_model=False)

def run(stub, spikes, state):
    stub.inputs['spikes'].copyin_numpy(np.ascontiguousarray(spikes, np.int32))
    stub.inputs['state_in'].copyin_numpy(np.ascontiguousarray(state, np.int32))
    before = len(audit_lines())
    stub.run()
    results = {name: value.copyout_numpy() for name, value in stub.outputs.items()}
    np.testing.assert_array_equal(stub.inputs['spikes'].copyout_numpy(), spikes)
    np.testing.assert_array_equal(stub.inputs['state_in'].copyout_numpy(), state)
    launches = audit_lines()[before:]
    return results, launches

def compare_case(name, spikes, wi, wr, state, parameters, output):
    model = sequence_model(len(spikes), wi, wr, parameters, trace_all=True)
    expected = numpy_reference(spikes, wi, wr, state, parameters)
    ref, _ = run(graph(reference_model(model)), spikes, state)
    qstub = graph(model)
    actual, launches = run(qstub, spikes, state)
    for key, value in zip(('features', 'state_out', 'trace'), expected):
        np.testing.assert_array_equal(ref[key], value, err_msg=name + ' CPU vs NumPy ' + key)
        np.testing.assert_array_equal(actual[key], value, err_msg=name + ' QEMU vs NumPy ' + key)
    assert len(launches) == 1 and launches[0]['status'] == 'ok'
    repeated, repeated_launches = run(qstub, spikes, state)
    for key in actual: np.testing.assert_array_equal(repeated[key], actual[key])
    assert len(repeated_launches) == 1
    np.savez_compressed(output / (name + '.npz'), spikes=spikes, initial_state=state,
                        features=actual['features'], state_out=actual['state_out'], trace=actual['trace'])
    return {'name': name, 'steps': len(spikes), 'neurons': wr.shape[0], 'trace_mismatches': 0,
            'launches_per_graph_run': 1, 'repeat_equal': True}

def tests(output):
    rng = np.random.default_rng(20261012)
    p = [17, 1, 7, 11, 2, 1, 5, 7, 3, 4, 1, 2, 0, 0, 0, 0, -2, 0, 1, 0, 0]
    spikes = rng.integers(0, 2, (19, 129), dtype=np.int32)
    wi = rng.integers(-8, 8, (129, 37), dtype=np.int32)
    wr = rng.integers(-8, 8, (37, 37), dtype=np.int32)
    state = np.zeros((5, 37), np.int32)
    state[:2] = rng.integers(-100, 100, (2, 37))
    state[2] = rng.integers(0, 4, 37)
    state[3] = rng.integers(0, 2, 37)
    state[4] = 10
    results = [compare_case('signed_tail_recurrent', spikes, wi, wr, state, p, output)]
    rounded = p.copy()
    rounded[12:16] = [1, 2, 2, 1]
    results.append(compare_case('nonzero_leakage_round_shifts', spikes, wi, wr, state, rounded, output))
    full, _ = run(graph(sequence_model(19, wi, wr, p, trace_all=True)), spikes, state)
    first, _ = run(graph(sequence_model(8, wi, wr, p, trace_all=True)), spikes[:8], state)
    second, _ = run(graph(sequence_model(11, wi, wr, p, trace_all=True)), spikes[8:], first['state_out'])
    np.testing.assert_array_equal(second['state_out'], full['state_out'])
    np.testing.assert_array_equal(second['features'], full['features'])
    np.testing.assert_array_equal(np.concatenate([first['trace'], second['trace']]), full['trace'])
    results.append({'name': 'chunked_state_continuation', 'trace_mismatches': 0})
    for mode in (0, 1):
        cp = [2, 0, 1, 1, 0, 0, 10, 10, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, mode]
        cs = np.ones((260, 8), np.int32)
        ci = np.full((8, 3), 2, np.int32)
        cr = np.zeros((3, 3), np.int32)
        results.append(compare_case('count_' + ('wide' if mode == 0 else 'uint8'), cs, ci, cr,
                                    np.zeros((5, 3), np.int32), cp, output))
    zp = [2, 0, 1000, 1000, 1, 2, 4, 4, 4, 4, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    results.append(compare_case('zero_input', np.zeros((11, 33), np.int32),
        np.zeros((33, 5), np.int32), np.zeros((5, 5), np.int32), np.zeros((5, 5), np.int32), zp, output))
    # Arithmetic wrap and shift extremes use an independent NumPy int64 boundary.
    wp = [1, 0, 2147483647, 2147483647, 0, 0, 31, 0, 31, 0, 31, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0]
    ws = np.zeros((5, 2), np.int32)
    ws[0] = [2147483647, -2147483648]; ws[1] = [-2147483648, 2147483647]
    results.append(compare_case('int32_wrap_shifts', np.ones((3, 1), np.int32),
        np.array([[7, -8]], np.int32), np.zeros((2, 2), np.int32), ws, wp, output))
    # Malformed model contracts fail before any process launch.
    invalid = sequence_model(19, wi, wr, p)
    invalid.graph.node[0].domain = 'wrong.domain'
    try: graph(invalid)
    except ValueError: pass
    else: raise AssertionError('Wrong domain accepted')
    invalid = sequence_model(19, wi, wr, p)
    invalid.opset_import[1].version = 2
    try: graph(invalid)
    except ValueError: pass
    else: raise AssertionError('Wrong custom opset version accepted')
    bad = spikes.copy(); bad[0, 0] = 2
    before = len(audit_lines())
    try: run(graph(sequence_model(19, wi, wr, p)), bad, state)
    except RuntimeError as error: assert 'binary' in str(error)
    else: raise AssertionError('Nonbinary input accepted')
    assert len(audit_lines()) == before
    bad_wi = wi.copy(); bad_wi[0, 0] = -9
    try: run(graph(sequence_model(19, bad_wi, wr, p)), spikes, state)
    except RuntimeError as error: assert 'INT4' in str(error)
    else: raise AssertionError('Out of range weight accepted')
    runner_before = os.environ['INFINI_RVNE_RUNNER']
    try:
        os.environ['INFINI_RVNE_RUNNER'] = '/bin/false'
        try: run(graph(sequence_model(19, wi, wr, p)), spikes, state)
        except RuntimeError as error: assert 'runner failed' in str(error)
        else: raise AssertionError('Failed runner accepted')
    finally: os.environ['INFINI_RVNE_RUNNER'] = runner_before
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        slow = Path(temporary) / 'slow_qemu.sh'
        slow.write_text('#!/bin/sh\nexec sleep 2\n'); slow.chmod(0o700)
        qemu_before = os.environ['INFINI_RVNE_QEMU']
        timeout_before = os.environ.get('INFINI_RVNE_TIMEOUT_MS')
        try:
            os.environ['INFINI_RVNE_QEMU'] = str(slow)
            os.environ['INFINI_RVNE_TIMEOUT_MS'] = '20'
            try: run(graph(sequence_model(19, wi, wr, p)), spikes, state)
            except RuntimeError as error: assert 'timeout' in str(error)
            else: raise AssertionError('Timeout not enforced')
        finally:
            os.environ['INFINI_RVNE_QEMU'] = qemu_before
            if timeout_before is None: os.environ.pop('INFINI_RVNE_TIMEOUT_MS', None)
            else: os.environ['INFINI_RVNE_TIMEOUT_MS'] = timeout_before
    results.append({'name': 'invalid_domain_version_spikes_weights_runner_timeout', 'status': 'PASS'})
    print('PASS synthetic and error tests', flush=True)
    return results

def material_validation(dataset, model_path, output, samples, policy):
    data = np.load(dataset, allow_pickle=False)
    model = onnx.load(model_path)
    # Primary comparison: identical LSMSequence -> MLP nodes on host and target.
    reference = onnx.load(Path(model_path).with_name('lsm_mlp_host.onnx'))
    assert [node.op_type for node in model.graph.node] == ['LSMSequence', 'MLP']
    assert reference.SerializeToString() == reference_model(model).SerializeToString()
    qgraph, rgraph = graph(model), graph(reference)
    state = np.zeros((5, data['recurrent_weights'].shape[0]), np.int32)
    rows, logits, features, states = [], [], [], []
    reference_logits, reference_features, reference_states = [], [], []
    for j in range(min(samples, len(data['labels']))):
        initial = state.copy() if policy == 'continuous' else np.zeros_like(state)
        initial[4] = 0  # Per-sample features, preserving dynamical state in continuous mode.
        spikes = data['spikes'][j]
        expected, host_launches = run(rgraph, spikes, initial)
        assert not host_launches, 'Host comparison launched QEMU'
        start = time.perf_counter()
        actual, launches = run(qgraph, spikes, initial)
        elapsed = time.perf_counter() - start
        assert len(launches) == 1 and launches[0]['status'] == 'ok'
        assert [r['command'] for r in launches[0]['requests']] == ['LSMSequence', 'MLP']
        for key in ('state_out', 'trace', 'features'):
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=f'sample {j}: {key}')
        np.testing.assert_allclose(actual['logits'], expected['logits'], rtol=1e-5, atol=1e-4)
        prediction = int(actual['logits'].argmax())
        assert prediction == int(expected['logits'].argmax())
        state = actual['state_out']
        rows.append({'sample': j, 'prediction': prediction, 'label': int(data['labels'][j]),
            'correct': prediction == int(data['labels'][j]), 'state_mismatches': 0,
            'logits_max_absolute_error': float(np.max(np.abs(actual['logits'] - expected['logits']))),
            'qemu_graph_wall_seconds': elapsed, 'launches': 1,
            'runner_pid': launches[0]['pid'], 'requests': launches[0]['requests'],
            'host_qemu_launches': len(host_launches)})
        logits.append(actual['logits']); features.append(actual['features']); states.append(state.copy())
        reference_logits.append(expected['logits']); reference_features.append(expected['features'])
        reference_states.append(expected['state_out'].copy())
        print(f'PASS sample={j} state_exact prediction={prediction} label={data["labels"][j]} seconds={elapsed:.3f}', flush=True)
    np.savez_compressed(output / ('material_outputs_' + policy + '.npz'), logits=np.concatenate(logits),
        features=np.concatenate(features), states=np.stack(states))
    np.savez_compressed(output / ('host_custom_outputs_' + policy + '.npz'),
        logits=np.concatenate(reference_logits), features=np.concatenate(reference_features),
        states=np.stack(reference_states))
    return {'policy': policy, 'samples': len(rows), 'correct': sum(r['correct'] for r in rows),
            'primary_reference': 'host InfiniTensor LSMSequence -> MLP, execution=reference',
            'accuracy': sum(r['correct'] for r in rows) / len(rows), 'results': rows,
            'scope': 'Fresh RVNE features on delivered cut events, C parameters and specified reset/count policy.'}

def checkpoint_validation(materials, outputs, report_path):
    import torch
    import torch.nn.functional as F
    data = np.load(outputs, allow_pickle=False)
    checkpoint = torch.load(materials / '7_best_model.pth', map_location='cpu', weights_only=True)
    with torch.inference_mode():
        x = torch.from_numpy(data['features'])
        hidden = torch.relu(F.linear(x, checkpoint['fc1.weight'], checkpoint['fc1.bias']))
        normalized = F.batch_norm(hidden, checkpoint['bn2.running_mean'], checkpoint['bn2.running_var'],
                                  checkpoint['bn2.weight'], checkpoint['bn2.bias'], training=False, eps=1e-5)
        expected = F.linear(normalized, checkpoint['fc4.weight'], checkpoint['fc4.bias']).numpy()
    np.testing.assert_allclose(data['logits'], expected, rtol=1e-5, atol=1e-4)
    np.testing.assert_array_equal(data['logits'].argmax(axis=1), expected.argmax(axis=1))
    report = {'status': 'PASS', 'samples': len(expected),
              'reference': 'original checkpoint: Linear -> ReLU -> BatchNorm(eval) -> Linear',
              'implementation': 'InfiniTensor MLP bridge -> RISC-V FP32 MLP; BN folded into following Linear',
              'max_absolute_error': float(np.max(np.abs(data['logits'] - expected))),
              'prediction_mismatches': 0, 'rtol': 1e-5, 'atol': 1e-4}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

def vendor_c_validation(materials, outputs, output):
    output.mkdir(parents=True, exist_ok=True)
    material = materials / 'c_80/c_80'
    original = (material / 'gesture_new.c').read_text()
    start = original.index('int mlp()')
    opening = original.index('{', start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (original[end] == '{') - (original[end] == '}')
        end += 1
    function = original[start:end]
    source = '''#include <math.h>
#include <string.h>
#include "config.h"
#include "mlp_weights.h"
float mlpinput[NeuronNum], mlphiden[hiden_layer], mlpoutput[output_layer];
float dps = 1e-5;
int classification_results;
''' + function + '''
void golden(const float *x, float *y) {
    memcpy(mlpinput, x, sizeof(mlpinput));
    mlp();
    memcpy(y, mlpoutput, sizeof(mlpoutput));
}
'''
    cfile = output / 'vendor_mlp_golden.c'
    library = output / 'vendor_mlp_golden.so'
    cfile.write_text(source)
    command = ['cc', '-std=c11', '-O2', '-ffp-contract=off', '-fPIC', '-shared',
               '-I' + str(material), str(cfile), '-lm', '-o', str(library)]
    subprocess.run(command, check=True, capture_output=True)
    lib = ctypes.CDLL(str(library.resolve()))
    pointer = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS')
    lib.golden.argtypes = [pointer, pointer]
    lib.golden.restype = None
    data = np.load(outputs, allow_pickle=False)
    features = data['features']
    expected = np.empty((len(features), 7), dtype=np.float32)
    for index, feature in enumerate(features):
        lib.golden(np.ascontiguousarray(feature, dtype=np.float32), expected[index])
    np.testing.assert_allclose(data['logits'], expected, rtol=1e-5, atol=1e-4)
    np.testing.assert_array_equal(data['logits'].argmax(1), expected.argmax(1))
    np.save(output / 'vendor_mlp_logits.npy', expected)
    summary = {'status': 'PASS', 'reference': 'verbatim gesture_new.c::mlp() + original mlp_weights.h',
               'samples': len(expected), 'max_absolute_error': float(np.max(np.abs(data['logits'] - expected))),
               'prediction_mismatches': 0, 'rtol': 1e-5, 'atol': 1e-4,
               'function_sha256': hashlib.sha256(function.encode()).hexdigest(),
               'source_sha256': hashlib.sha256((material / 'gesture_new.c').read_bytes()).hexdigest(),
               'weights_sha256': hashlib.sha256((material / 'mlp_weights.h').read_bytes()).hexdigest(),
               'build_command': command,
               'scope': 'MLP conditional on identical features; original C retains explicit BatchNorm.',
               'full_vendor_driver_is_golden': False,
               'full_driver_differences': ['continuous dynamical state across samples',
                   '8-bit hardware count instead of default wide software count',
                   'hardcoded event count and boundary check order',
                   'original propagation and per-step state-clearing behavior require separate alignment']}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

def mlp_model(x, weights, execution='qemu', standard=False):
    nodes = [helper.make_node('MLP', ['x', 'W1', 'b1', 'W2', 'b2'], ['y'],
                             domain=DOMAIN, execution=execution)]
    if standard:
        nodes = [helper.make_node('MatMul', ['x', 'W1'], ['a']),
                 helper.make_node('Add', ['a', 'b1'], ['b']),
                 helper.make_node('Relu', ['b'], ['c']),
                 helper.make_node('MatMul', ['c', 'W2'], ['d']),
                 helper.make_node('Add', ['d', 'b2'], ['y'])]
    model = helper.make_model(helper.make_graph(nodes, 'MLP_test',
        [helper.make_tensor_value_info('x', TensorProto.FLOAT, list(x.shape))],
        [helper.make_tensor_value_info('y', TensorProto.FLOAT, [x.shape[0], weights['b2'].size])],
        [numpy_helper.from_array(value, key) for key, value in weights.items()]),
        opset_imports=[helper.make_opsetid('', 13), helper.make_opsetid(DOMAIN, 1)])
    model.ir_version = 8
    return model

def mlp_run(model, x):
    stub = graph(model)
    stub.inputs['x'].copyin_numpy(np.ascontiguousarray(x, dtype=np.float32))
    before = len(audit_lines())
    stub.run()
    return stub.outputs['y'].copyout_numpy(), audit_lines()[before:]

def session_validation(output):
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261021)
    tests = []
    for batch, d, h, o in [(1, 1, 1, 1), (3, 7, 11, 5), (1, 80, 496, 7)]:
        x = rng.normal(size=(batch, d)).astype(np.float32)
        weights = {k: rng.normal(scale=.2, size=s).astype(np.float32)
                   for k, s in [('W1', (d, h)), ('b1', (h,)), ('W2', (h, o)), ('b2', (o,))]}
        model = mlp_model(x, weights)
        expected, events = mlp_run(mlp_model(x, weights, standard=True), x)
        assert not events
        actual, events = mlp_run(model, x)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-4)
        assert len(events) == 1 and events[0]['status'] == 'ok'
        assert [r['command'] for r in events[0]['requests']] == ['MLP']
        tests.append({'name': f'mlp_{batch}_{d}_{h}_{o}', 'max_absolute_error': float(np.max(np.abs(actual-expected)))})
    # Real protocol path: malformed request rejected, same runner accepts next request.
    command = [os.environ['INFINI_RVNE_QEMU'], '-L', os.environ['INFINI_RVNE_SYSROOT'],
               os.environ['INFINI_RVNE_RUNNER'], '--serve']
    def packet(cmd, request_id, payload):
        return struct.pack('<6I', 0x52565131, 2, cmd, request_id, len(payload), 0) + payload
    bad = struct.pack('<4I', 1, 0, 1, 1)
    valid = struct.pack('<4I', 1, 1, 1, 1) + np.array([2, 3, 1, 4, 5], '<f4').tobytes()
    response = subprocess.run(command, input=packet(2, 1, bad) + packet(2, 2, valid),
                              capture_output=True, check=True, timeout=30).stdout
    offset = 0
    for request_id in (1, 2):
        header = struct.unpack_from('<6I', response, offset); offset += 24
        assert header[:4] == (0x52565132, 2, 2, request_id)
        payload = response[offset:offset+header[4]]; offset += header[4]
        if request_id == 1: assert header[5] == 1 and b'dimensions' in payload
        else:
            assert header[5] == 0
            np.testing.assert_array_equal(np.frombuffer(payload, '<f4'), [33])
    assert offset == len(response)
    tests.append({'name': 'target_error_response_and_next_request', 'status': 'PASS'})
    for payload in (b'bad', struct.pack('<6I', 0x52565131, 99, 2, 1, 0, 0),
                    packet(2, 1, valid)[:-2]):
        failed = subprocess.run(command, input=payload, capture_output=True, timeout=30)
        assert failed.returncode != 0
    tests.append({'name': 'target_truncated_frame_and_bad_version', 'status': 'PASS'})
    # Both nodes, host reference, replay and state continuation.
    d, n, t = 9, 5, 13
    spikes = rng.integers(0, 2, (t, d), dtype=np.int32)
    wi = rng.integers(-8, 8, (d, n), dtype=np.int32)
    wr = rng.integers(-8, 8, (n, n), dtype=np.int32)
    state = np.zeros((5, n), np.int32)
    p = [3, 0, 7, 9, 1, 2, 5, 5, 5, 5, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    weights = {k: rng.normal(scale=.2, size=s).astype(np.float32)
               for k, s in [('W1', (n, 11)), ('b1', (11,)), ('W2', (11, 3)), ('b2', (3,))]}
    model = sequence_model(t, wi, wr, p, mlp=weights, trace_all=True)
    assert [node.op_type for node in model.graph.node] == ['LSMSequence', 'MLP']
    expected, reference_events = run(graph(reference_model(model)), spikes, state)
    assert not reference_events
    stub = graph(model)
    for _ in range(2):
        actual, events = run(stub, spikes, state)
        for key in ('features', 'state_out', 'trace'):
            np.testing.assert_array_equal(actual[key], expected[key])
        np.testing.assert_allclose(actual['logits'], expected['logits'], rtol=1e-5, atol=1e-4)
        assert len(events) == 1 and events[0]['status'] == 'ok'
        assert [r['command'] for r in events[0]['requests']] == ['LSMSequence', 'MLP']
        assert all(r['status'] == 'ok' for r in events[0]['requests'])
        try: os.kill(events[0]['pid'], 0)
        except ProcessLookupError: pass
        else: raise AssertionError('QEMU child not reaped')
    first, _ = run(graph(sequence_model(5, wi, wr, p, mlp=weights)), spikes[:5], state)
    second, events = run(graph(sequence_model(t-5, wi, wr, p, mlp=weights)), spikes[5:], first['state_out'])
    np.testing.assert_array_equal(second['state_out'], expected['state_out'])
    np.testing.assert_allclose(second['logits'], expected['logits'], rtol=1e-5, atol=1e-4)
    tests.append({'name': 'two_nodes_one_pid_replay_continuation_and_reaping', 'status': 'PASS'})
    # Check malformed custom contracts and invalid FP32 data before target work.
    for mutation in ('arity', 'execution', 'domain', 'dtype', 'shape'):
        broken = onnx.ModelProto.FromString(model.SerializeToString())
        node = broken.graph.node[1]
        if mutation == 'arity': node.input.pop()
        if mutation == 'execution': node.attribute[0].s = b'bad'
        if mutation == 'domain': node.domain = 'wrong'
        if mutation in ('dtype', 'shape'):
            for item in broken.graph.initializer:
                if item.name == 'W1':
                    value = onnx.numpy_helper.to_array(item).copy()
                    value = value.astype(np.int32) if mutation == 'dtype' else value[:-1]
                    item.CopyFrom(numpy_helper.from_array(value, 'W1'))
        try: graph(broken)
        except (RuntimeError, ValueError): pass
        else: raise AssertionError('Invalid MLP contract accepted: ' + mutation)
    tests.append({'name': 'invalid_mlp_contracts', 'status': 'PASS'})
    # Host detects corrupt replies, child death and mid-frame timeout, then recovers.
    x = np.ones((1, 1), np.float32)
    weights = {k: np.ones(s, np.float32) for k, s in [('W1', (1, 1)), ('b1', (1,)), ('W2', (1, 1)), ('b2', (1,))]}
    for value in (np.nan, np.inf):
        try: mlp_run(mlp_model(x, weights), np.full_like(x, value))
        except RuntimeError as error: assert 'non-finite' in str(error)
        else: raise AssertionError('Nonfinite input accepted')
    with tempfile.TemporaryDirectory(dir=output) as directory:
        mock = Path(directory) / 'mock_qemu'
        saved_qemu = os.environ['INFINI_RVNE_QEMU']
        saved_timeout = os.environ.get('INFINI_RVNE_TIMEOUT_MS')
        try:
            for mode in ('version', 'length', 'truncated', 'timeout'):
                mock.write_text('''#!/usr/bin/env python3
import sys,struct,time
h=sys.stdin.buffer.read(24)
magic,v,c,i,n,s=struct.unpack('<6I',h)
sys.stdin.buffer.read(n)
mode=''' + repr(mode) + '''
if mode=='version': response=struct.pack('<6I',0x52565132,99,c,i,4,0)
elif mode=='length': response=struct.pack('<6I',0x52565132,2,c,i,999999999,0)
elif mode=='truncated': response=b'abc'
else:
    time.sleep(2)
    response=b''
sys.stdout.buffer.write(response)
sys.stdout.buffer.flush()
''')
                mock.chmod(0o700)
                os.environ['INFINI_RVNE_QEMU'] = str(mock)
                os.environ['INFINI_RVNE_TIMEOUT_MS'] = '500'
                before = len(audit_lines())
                try: mlp_run(mlp_model(x, weights), x)
                except RuntimeError: pass
                else: raise AssertionError('Corrupt runner accepted: ' + mode)
                events = audit_lines()[before:]
                assert len(events) == 1 and events[0]['status'] != 'ok'
                try: os.kill(events[0]['pid'], 0)
                except ProcessLookupError: pass
                else: raise AssertionError('Failed child not reaped')
        finally:
            os.environ['INFINI_RVNE_QEMU'] = saved_qemu
            if saved_timeout is None: os.environ.pop('INFINI_RVNE_TIMEOUT_MS', None)
            else: os.environ['INFINI_RVNE_TIMEOUT_MS'] = saved_timeout
    actual, events = mlp_run(mlp_model(x, weights), x)
    np.testing.assert_array_equal(actual, [[3]])
    assert len(events) == 1 and events[0]['status'] == 'ok'
    tests.append({'name': 'host_corrupt_reply_timeout_cleanup_and_recovery', 'status': 'PASS'})
    report = {'status': 'PASS', 'tests': tests, 'two_nodes_per_graph': True,
              'qemu_launches_per_graph_run': 1, 'feature_transfer': 'host tensor roundtrip',
              'runner_lifetime': 'one CPU graph run; no leaked process across runs'}
    (output / 'summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--materials', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=25)
    parser.add_argument('--policy', choices=['independent', 'continuous'], default='independent')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {'status': 'RUNNING', 'backend_module': str(backend.__file__)}
    try:
        summary['tests'] = tests(args.output_dir)
        summary['material'] = material_validation(args.dataset, args.model, args.output_dir, args.samples, args.policy)
        # A real-material prefix also has full per-step differential evidence.
        data = np.load(args.dataset, allow_pickle=False)
        summary['material_prefix_trace'] = compare_case('material_prefix_128', data['spikes'][0, :128],
            data['input_weights'], data['recurrent_weights'], np.zeros((5, 80), np.int32),
            data['parameters'].tolist(), args.output_dir)
        outputs = args.output_dir / ('material_outputs_' + args.policy + '.npz')
        checkpoint_validation(args.materials, outputs, args.output_dir / 'readout_summary.json')
        vendor_c_validation(args.materials, outputs, args.output_dir.parent / 'vendor_c_golden')
        session_validation(args.output_dir.parent / 'session_validation')
        summary['status'] = 'PASS'
    except Exception as error:
        summary['status'] = 'FAIL'; summary['error'] = str(error)
        raise
    finally:
        (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({'status': summary['status'], 'material_samples': summary['material']['samples'],
                     'accuracy': summary['material']['accuracy'], 'test_cases': len(summary['tests'])}, indent=2))
if __name__ == '__main__': main()
