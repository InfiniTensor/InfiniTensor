"""Lower dense two-operand contractions to Transpose, Reshape, and MatMul."""

from math import prod
from string import ascii_letters


def lower_einsum(handler, equation, lhs, rhs, output, activation, compute_type):
    equation = "".join(equation.split())
    operands, separator, result = equation.partition("->")
    terms = operands.split(",")
    if len(terms) != 2 or any(c not in ascii_letters for c in "".join(terms) + result):
        raise NotImplementedError(f"Unsupported Einsum equation: {equation}")
    left, right = terms
    if len(set(left)) != len(left) or len(set(right)) != len(right):
        raise NotImplementedError("Einsum diagonals are not supported")
    if not separator:
        result = "".join(sorted(set(left) ^ set(right)))
    if len(set(result)) != len(result) or any(c not in left + right for c in result):
        raise ValueError(f"Invalid Einsum output: {equation}")
    batch = [c for c in result if c in left and c in right]
    contract = [c for c in left if c in right and c not in result]
    free_left = [c for c in left if c not in right]
    free_right = [c for c in right if c not in left]
    if any(c not in result for c in free_left + free_right):
        raise NotImplementedError("Einsum standalone reductions are not supported")

    lhs_shape, rhs_shape = lhs.shape(), rhs.shape()
    if len(left) != len(lhs_shape) or len(right) != len(rhs_shape):
        raise ValueError("Einsum labels must match tensor ranks")
    sizes_l, sizes_r = dict(zip(left, lhs_shape)), dict(zip(right, rhs_shape))
    if any(dim <= 0 for dim in (*lhs_shape, *rhs_shape)):
        raise NotImplementedError("Einsum requires positive concrete dimensions")
    if any(sizes_l[c] != sizes_r[c] for c in contract):
        raise ValueError("Einsum contraction dimensions do not match")
    if any(sizes_l[c] != sizes_r[c] and min(sizes_l[c], sizes_r[c]) != 1 for c in batch):
        raise ValueError("Einsum batch dimensions do not broadcast")

    def ordered_view(tensor, original, order, shape):
        permutation = [original.index(c) for c in order]
        if permutation != list(range(len(original))):
            tensor = handler.transpose(tensor, None, permutation)
        return handler.reshape(tensor, None, shape) if tensor.shape() != shape else tensor

    rows = prod(sizes_l[c] for c in free_left)
    inner = prod(sizes_l[c] for c in contract)
    columns = prod(sizes_r[c] for c in free_right)
    lhs_view = ordered_view(lhs, left, batch + free_left + contract,
                            [sizes_l[c] for c in batch] + [rows, inner])
    rhs_view = ordered_view(rhs, right, batch + contract + free_right,
                            [sizes_r[c] for c in batch] + [inner, columns])
    canonical = batch + free_left + free_right
    canonical_shape = ([max(sizes_l[c], sizes_r[c]) for c in batch]
                       + [sizes_l[c] for c in free_left] + [sizes_r[c] for c in free_right])
    product = handler.matmul(lhs_view, rhs_view, None, False, False, None,
                             activation, compute_type)
    permutation = [canonical.index(c) for c in result]
    if permutation != list(range(len(result))):
        shaped = handler.reshape(product, None, canonical_shape)
        return handler.transpose(shaped, output, permutation)
    return handler.reshape(product, output, canonical_shape)
