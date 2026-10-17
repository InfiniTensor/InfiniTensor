"""Ensure captured ATen temporaries cannot alias unrelated live tensors."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from pyinfinitensor.embedding import EmbeddingSession
p=argparse.ArgumentParser();p.add_argument('--model-dir');p.add_argument('--device');p.add_argument('--output');a=p.parse_args()
device={'nvidia':'cuda','cambricon':'mlu','ascend':'npu'}[a.device]
s=EmbeddingSession(a.model_dir,device=a.device,execution_mode='graph')
texts=['中'*126]*8
reference=s.encode(texts)
first=s.encode(texts)
np.testing.assert_allclose(first,reference,atol=1e-4,rtol=1e-3)
sentinels=[]
for dtype in (torch.float32,torch.int64,torch.bool):
 for elements in (128,1024,4096,16384,65536,262144):
  for i in range(8):
   value=True if dtype==torch.bool else 37+i
   sentinels.append((torch.full((elements,),value,dtype=dtype,device=device),value))
for i in range(5):
 actual=s.encode(['文'*126]*8 if i%2 else texts)
 for tensor,value in sentinels:
  assert bool(torch.all(tensor==value).item()), 'Captured temporary overwrote an unrelated live tensor'
assert s.execution_stats()['captures']==1
Path(a.output).write_text(json.dumps({'status':'PASS','sentinels':len(sentinels),'execution':s.execution_stats()},indent=2))
print('Capture memory isolation PASS',a.device)
