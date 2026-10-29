"""Deterministic, exact-length batching without buckets or static models."""
import operator
from typing import List, Sequence


def positive_integer(value, name):
    if isinstance(value, bool):
        raise ValueError(name + ' must be a positive integer')
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(name + ' must be a positive integer') from error
    if result < 1:
        raise ValueError(name + ' must be a positive integer')
    return result


def plan_batches(lengths: Sequence[int], batch_size: int, token_budget: int,
                 order: str = 'length') -> List[List[int]]:
    """Group similar lengths; pad only to each batch's actual maximum.

    Returned indices preserve stable ordering among equal lengths. Consumers
    scatter outputs back to the caller's original order.
    """
    batch_size = positive_integer(batch_size, 'batch_size')
    token_budget = positive_integer(token_budget, 'token_budget')
    if order not in ('input', 'length'):
        raise ValueError('batching must be input or length')
    sizes = [positive_integer(value, 'sequence length') for value in lengths]
    if sizes and max(sizes) > token_budget:
        raise ValueError('token_budget cannot fit the longest sequence')
    indices = sorted(range(len(sizes)), key=lambda index: (sizes[index], index)) if order == 'length' else list(range(len(sizes)))
    batches, current, longest = [], [], 0
    for index in indices:
        candidate_max = max(longest, sizes[index])
        if current and (len(current) == batch_size or (len(current) + 1) * candidate_max > token_budget):
            batches.append(current)
            current, longest = [], 0
        current.append(index)
        longest = max(longest, sizes[index])
    if current:
        batches.append(current)
    return batches
