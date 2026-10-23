"""Standalone ONNX inference: InfiniTensor host graph calls the RVNE runner."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import onnx
from pyinfinitensor import backend
from pyinfinitensor.onnx import OnnxStub

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--sample', type=int, default=0)
    parser.add_argument('--state-in', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    data = np.load(args.dataset, allow_pickle=False)
    spikes = np.ascontiguousarray(data['spikes'][args.sample], np.int32)
    neurons = data['recurrent_weights'].shape[0]
    state = np.load(args.state_in, allow_pickle=False) if args.state_in else np.zeros((5, neurons), np.int32)
    graph = OnnxStub(onnx.load(args.model), backend.cpu_runtime())
    graph.inputs['spikes'].copyin_numpy(spikes)
    graph.inputs['state_in'].copyin_numpy(np.ascontiguousarray(state, np.int32))
    graph.run()
    result = {name: value.copyout_numpy() for name, value in graph.outputs.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_dir / 'outputs.npz', **result)
    np.save(args.output_dir / 'state_out.npy', result['state_out'])
    prediction = int(result['logits'].argmax())
    summary = {'sample': args.sample, 'prediction': prediction,
        'prediction_angle': float(data['classes'][prediction]), 'label': int(data['labels'][args.sample]),
        'operators': [str(op.op_type().id()) for op in graph.handler.operators()],
        'torch_imported': 'torch' in sys.modules, 'outputs': {k: list(v.shape) for k, v in result.items()}}
    assert not summary['torch_imported'], 'Inference unexpectedly imported Torch'
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
if __name__ == '__main__': main()
