#!/usr/bin/env python3
"""Transfer an identical ONNX package without duplicating public reference weights.

The small template preserves protobuf fields verbatim. Restore is accepted only
when the reconstructed ONNX matches the frozen full-file SHA256.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import onnx
from onnx import numpy_helper
from safetensors.numpy import load_file

from accept_m1 import sha256
import hashlib


def array_digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['pack','restore'])
    parser.add_argument('--source',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--weights',help='verified local model.safetensors, required for restore')
    args=parser.parse_args()
    source,output=Path(args.source),Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((source/'manifest.json').read_text())
    weight_path=source/'model.safetensors' if args.command=='pack' else Path(args.weights)
    if sha256(weight_path)!=manifest['sha256']['model.safetensors']:
        raise ValueError('reference weights differ from frozen manifest')
    weights=load_file(str(weight_path))
    for path in source.iterdir():
        if path.is_file() and path.name not in ('model.onnx','model.safetensors','template.onnx','template-map.json'):
            shutil.copyfile(path,output/path.name)
    if args.command=='pack':
        assert sha256(source/'model.onnx')==manifest['sha256']['model.onnx']
        candidates={}
        for name,value in weights.items():
            candidates[(tuple(value.shape),array_digest(value))]=(name,False)
            if value.ndim==2:
                candidates[(tuple(value.T.shape),array_digest(value.T))]=(name,True)
        model=onnx.load(str(source/'model.onnx'));bindings=[]
        for index,tensor in enumerate(model.graph.initializer):
            if not tensor.raw_data or len(tensor.raw_data)<4096:continue
            value=numpy_helper.to_array(tensor)
            match=candidates.get((tuple(value.shape),array_digest(value)))
            if match:
                bindings.append({'index':index,'weight':match[0],'transpose':match[1]})
                tensor.ClearField('raw_data')
        (output/'template.onnx').write_bytes(model.SerializeToString())
        (output/'template-map.json').write_text(json.dumps(bindings))
        print(json.dumps({'bindings':len(bindings),'template_bytes':(output/'template.onnx').stat().st_size}))
    else:
        model=onnx.load(str(source/'template.onnx'))
        for item in json.loads((source/'template-map.json').read_text()):
            value=weights[item['weight']]
            if item['transpose']:value=value.T
            tensor=model.graph.initializer[item['index']]
            assert list(value.shape)==list(tensor.dims)
            tensor.raw_data=np.ascontiguousarray(value).tobytes()
        (output/'model.onnx').write_bytes(model.SerializeToString())
        if sha256(output/'model.onnx')!=manifest['sha256']['model.onnx']:
            raise ValueError('reconstructed ONNX differs from the frozen package')
        # Reference weight file is immutable and independently verified.
        try:
            (output/'model.safetensors').hardlink_to(weight_path.resolve())
        except OSError:
            shutil.copyfile(weight_path,output/'model.safetensors')
        print(json.dumps({'status':'PASS','model_onnx_sha256':manifest['sha256']['model.onnx']}))


if __name__=='__main__':main()
