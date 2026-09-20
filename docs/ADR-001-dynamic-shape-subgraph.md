# ADR-001: Runtime Shape Subgraph Preparation

## Status

Accepted for the ONNX dynamic Shape project.

## Decision

An ONNX graph with runtime-dependent shape values is executed in three ordered
phases:

1. Update the concrete input shapes and validate fixed dimensions.
2. Execute the shape-producing subgraph (`Shape`, `Gather`, `Unsqueeze`,
   `Squeeze`, `Concat`, and `Cast`) on the selected runtime.
3. Read dynamic shape tensors through the operator shape-inference path, call
   `Graph::shape_infer()`, and re-run `Graph::dataMalloc()` before executing the
   complete data graph.

Dynamic `Reshape` keeps its legacy static-shape constructor and adds a second
`Tensor` input. Until phase 2 has produced a value, its output uses the input
shape as a temporary construction shape. After phase 2, ONNX `0` and `-1`
semantics are resolved against the current input and invalid element counts are
rejected.

## Rationale

The existing allocator already handles shape changes, activation storage
replacement, weight preservation, and CUDA graph invalidation. Keeping shape
preparation before data execution reuses those guarantees and avoids changing
memory layout while a data kernel is running. A full symbolic expression system
is unnecessary for the supported ONNX subset.

## Compatibility

Static ONNX `Reshape` remains unchanged. Static input models continue through
the existing graph execution path. Dynamic dimension metadata is kept in the
Python importer, while `TensorObj` continues to hold the concrete shape for the
current execution.

## Known limits

The first milestone implements CPU execution of the shape phase. CUDA shape
kernel registration and CUDA dynamic-shape execution are separate milestones;
they must be validated with device-side shape values and CUDA graph recapture
before being claimed as complete. `ConstantOfShape`, dynamic axes, and ONNX
`allowzero` are outside this first slice.
