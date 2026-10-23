"""Export the delivered quantized reservoir and trained readout to QEMU and host ONNX models."""
import argparse
import copy
from onnx import TensorProto, helper, numpy_helper
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import onnx

DOMAIN = 'com.infinitensor.rvne'
PARAMETER_NAMES = ['excitatory_count', 'weight_shift', 'threshold_inhibit', 'threshold_excite',
    'period_inhibit', 'period_excite', 'voltage_shift_inhibit', 'voltage_shift_excite',
    'current_shift_inhibit', 'current_shift_excite', 'vi_shift_inhibit', 'vi_shift_excite',
    'round_voltage_inhibit', 'round_voltage_excite', 'round_current_inhibit', 'round_current_excite',
    'reset_voltage_inhibit', 'reset_voltage_excite', 'reset_current_inhibit', 'reset_current_excite',
    'count_mode']
DEFAULT_PARAMETERS = [64, 9, 12672, 11520, 20, 50, 5, 7, 8, 8, 6, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0]
STATE_NAMES = ['current', 'voltage', 'refractory', 'previous_spike', 'count']

def sequence_model(steps, wi, wr, parameters, execution='qemu', trace_all=False, mlp=None):
    channels, neurons = wi.shape
    wi, wr = np.asarray(wi, np.int32), np.asarray(wr, np.int32)
    nodes = [helper.make_node('LSMSequence', ['spikes', 'input_weights', 'recurrent_weights', 'state_in'],
        ['features', 'state_out', 'trace'], domain=DOMAIN, name='lsm_sequence',
        parameters=list(map(int, parameters)), execution=execution, trace_all=int(trace_all))]
    initializers = [numpy_helper.from_array(wi, 'input_weights'), numpy_helper.from_array(wr, 'recurrent_weights')]
    inputs = [helper.make_tensor_value_info('spikes', TensorProto.INT32, [steps, channels]),
              helper.make_tensor_value_info('state_in', TensorProto.INT32, [5, neurons])]
    outputs = [helper.make_tensor_value_info('features', TensorProto.FLOAT, [1, neurons]),
               helper.make_tensor_value_info('state_out', TensorProto.INT32, [5, neurons]),
               helper.make_tensor_value_info('trace', TensorProto.INT32, [steps if trace_all else 1, 5, neurons])]
    if mlp is not None:
        for name in ('W1', 'b1', 'W2', 'b2'):
            initializers.append(numpy_helper.from_array(np.ascontiguousarray(mlp[name], np.float32), name))
        nodes.append(helper.make_node('MLP', ['features', 'W1', 'b1', 'W2', 'b2'],
            ['logits'], domain=DOMAIN, name='mlp_readout', execution=execution))
        outputs.insert(0, helper.make_tensor_value_info('logits', TensorProto.FLOAT, [1, mlp['b2'].size]))
    model = helper.make_model(helper.make_graph(nodes, 'LSM_QEMU_Readout', inputs, outputs, initializers),
        opset_imports=[helper.make_opsetid('', 13), helper.make_opsetid(DOMAIN, 1)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model

def reference_model(model):
    result = copy.deepcopy(model)
    for node in result.graph.node:
        if node.domain == DOMAIN and node.op_type in ('LSMSequence', 'MLP'):
            for attr in node.attribute:
                if attr.name == 'execution': attr.s = b'reference'
    return result

def numpy_reference(spikes, wi, wr, initial, p):
    """Independent vectorized implementation; all arithmetic wraps at int32."""
    s = np.array(initial, dtype=np.int32, copy=True)
    n = wr.shape[0]
    excited = np.arange(n) < p[0]
    choose = lambda a, b: np.where(excited, p[b], p[a]).astype(np.int64)
    threshold, period = choose(2, 3), choose(4, 5)
    vs, cs, vi = choose(6, 7), choose(8, 9), choose(10, 11)
    rv, ri = choose(12, 13), choose(14, 15)
    resetv, reseti = choose(16, 17), choose(18, 19)
    wi64, wr64 = np.asarray(wi, np.int64), np.asarray(wr, np.int64)
    wrap = lambda x: np.asarray(x).astype(np.int32).astype(np.int64)
    trace = []
    for sample in spikes:
        current = wrap(s[0].astype(np.int64) +
            (np.asarray(sample, np.int64) @ wi64 + s[3].astype(np.int64) @ wr64) * (1 << p[1]))
        voltage = s[1].astype(np.int64)
        refractory = s[2].astype(np.int64)
        ni = wrap(current - ((current >> cs) >> ri))
        nv = np.where(refractory > 0, voltage,
            wrap(voltage - ((voltage >> vs) >> rv) + (current >> vi)))
        fired = (refractory == 0) & (nv > threshold)
        count = s[4].astype(np.int64) + fired
        if p[20]: count &= 255
        s = np.stack((np.where(fired, reseti, ni), np.where(fired, resetv, nv),
            np.where(fired, period, np.maximum(refractory - 1, 0)), fired.astype(np.int64), count)).astype(np.int32)
        trace.append(s.copy())
    features = np.zeros(n, np.float32)
    for start, end in ((0, p[0]), (p[0], n)):
        maximum = s[4, start:end].max(initial=0)
        if maximum: features[start:end] = s[4, start:end].astype(np.float32) / np.float32(maximum)
    return features[None], s, np.stack(trace)


def main():
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('--materials', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--count-mode', choices=['wide', 'uint8'], default='wide')
    args = parser.parse_args()
    root, out = args.materials, args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(root / '7_best_model.pth', map_location='cpu', weights_only=True)
    with torch.inference_mode():
        alpha = checkpoint['bn2.weight'] / torch.sqrt(checkpoint['bn2.running_var'] + 1e-5)
        beta = checkpoint['bn2.bias'] - alpha * checkpoint['bn2.running_mean']
        mlp = {'W1': checkpoint['fc1.weight'].T.contiguous().numpy(), 'b1': checkpoint['fc1.bias'].numpy(),
            'W2': (checkpoint['fc4.weight'] * alpha).T.contiguous().numpy(),
            'b2': (checkpoint['fc4.bias'] + checkpoint['fc4.weight'] @ beta).numpy()}
    wi = np.load(root / 'weight_processed/q_weight_IN.npy').astype(np.int32)
    wr = np.load(root / 'weight_processed/q_weight_LSM.npy').astype(np.int32)
    assert wi.shape == (200, 80) and wr.shape == (80, 80)
    assert wi.min() >= -8 and wi.max() <= 7 and wr.min() >= -8 and wr.max() <= 7
    # Match the supplied C fixture's integer timestep conversion exactly.
    # The original event times are converted with truncation after multiplying by 10.
    times = np.load(root / 'spikes_input/3second_7direction/time_test_cut.npy')
    indices = np.load(root / 'spikes_input/3second_7direction/index_test_cut.npy').astype(np.int64)
    order = np.argsort(times, kind='stable')
    times, indices = times[order], indices[order]
    ticks = (times * 10).astype(np.int64)
    header = (root / 'c_80/c_80/input_arrays.h').read_text()
    # C initializers have a trailing comma; np.fromstring may append an extra
    # value for trailing whitespace after that comma.
    delivered_ticks = np.array(re.findall(r'-?\d+', re.search(r'TIMES\[\d+\]\s*=\s*\{([^}]+)\}', header, re.S)[1]), dtype=np.int64)
    delivered_indices = np.array(re.findall(r'-?\d+', re.search(r'INDICES\[\d+\]\s*=\s*\{([^}]+)\}', header, re.S)[1]), dtype=np.int64)
    assert np.array_equal(ticks, delivered_ticks), 'Event conversion differs from supplied C fixture'
    # Ordering within a timestep is irrelevant to binary spike packing. Compare
    # the complete event multiset so that duplicate events are checked as well.
    assert np.array_equal(np.sort(ticks * 200 + indices),
                          np.sort(delivered_ticks * 200 + delivered_indices)), 'Events differ from supplied C fixture'
    labels = np.load(root / 'spikes_input/3second_7direction/label_test_cut.npy')
    classes = np.unique(np.load(root / 'spikes_input/3second_7direction/train_label.npy'))
    encoded_labels = np.searchsorted(classes, labels).astype(np.int32)
    assert np.array_equal(classes[encoded_labels], labels)
    samples, steps = len(labels), 6000
    assert np.all((ticks >= 0) & (ticks < samples * steps)) and np.all((indices >= 0) & (indices < 200))
    spikes = np.zeros((samples, steps, 200), np.int32)
    spikes[ticks // steps, ticks % steps, indices] = 1
    parameters = list(DEFAULT_PARAMETERS)
    parameters[-1] = int(args.count_mode == 'uint8')
    model = sequence_model(steps, wi, wr, parameters, mlp=mlp)
    onnx.save(model, out / 'lsm_mlp_qemu.onnx')
    onnx.save(reference_model(model), out / 'lsm_mlp_host.onnx')
    np.savez_compressed(out / 'dataset.npz', spikes=spikes, labels=encoded_labels, classes=classes,
        input_weights=wi, recurrent_weights=wr, parameters=np.array(parameters, np.int32), **mlp)
    manifest = {'contract_version': 1, 'execution': 'two host-dispatched nodes, one persistent QEMU per graph run', 'steps': steps, 'samples': samples, 'channels': 200, 'neurons': 80,
        'parameters': dict(zip(PARAMETER_NAMES, parameters)), 'state_rows': STATE_NAMES,
        'time_conversion': 'stable time sort, then int64(original_ms * 10), verified against input_arrays.h',
        'duplicate_events': 'same timestep/channel collapsed to one binary spike, matching C bit packing',
        'label_mapping': classes.tolist(), 'default_sample_policy': 'independent zero initial states',
        'models': {name: {'execution': execution,
                         'sha256': hashlib.sha256((out / name).read_bytes()).hexdigest()}
                   for name, execution in [('lsm_mlp_qemu.onnx', 'qemu'), ('lsm_mlp_host.onnx', 'reference')]},
        'checkpoint_sha256': hashlib.sha256((root / '7_best_model.pth').read_bytes()).hexdigest(),
        'original_events': len(ticks), 'binary_spikes': int(spikes.sum()),
        'note': 'Uses C inference parameters; does not claim reproducing Brian2 training or archived LSM features.'}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
if __name__ == '__main__': main()
