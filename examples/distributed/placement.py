from typing import Optional


class Placement:
    # base class Placement type

    # convenient utils to check for placement types
    def is_shard(self, dim: Optional[int] = None) -> bool:
        if dim is not None and isinstance(self, Shard):
            return self.dim == dim
        else:
            return isinstance(self, Shard)

    def is_replicate(self) -> bool:
        return isinstance(self, Replicate)

    def is_partial(self) -> bool:
        return isinstance(self, _Partial)


class Replicate(Placement):
    def __repr__(self) -> str:
        """
        machine readable representation of the Replicate placement
        """
        return "Replicate()"


class Shard(Placement):
    # shard placement, shard on a dim
    def __init__(self, dim):
        self.dim = dim

    def __repr__(self) -> str:
        """
        machine readable representation of the Shard placement
        """
        return f"Shard(dim={self.dim})"


class _Partial(Placement):
    def __init__(self, reduce_op: str = "sum"):
        self.reduce_op: str = reduce_op

    def __repr__(self) -> str:
        """
        machine readable representation of the Partial placement
        """
        return f"_Partial(reduce_op={self.reduce_op})"
