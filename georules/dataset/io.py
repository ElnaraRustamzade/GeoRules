"""Per-rank shard writing for the dataset generator.

Each rank owns its own shard namespace,
``output_dir/shard_r{rank:04d}_s{shard_idx:06d}``, so no coordination
between ranks is needed. Shards are assembled in a ``.tmp`` sibling
directory and atomically renamed into place, so a shard is either
complete or absent.
"""

import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


class ShardWriter:
    """Buffer samples in memory; flush to disk in shards of fixed size."""

    def __init__(self, output_dir: str, rank: int, shard_size: int):
        self.output_dir = Path(output_dir)
        self.rank = int(rank)
        self.shard_size = int(shard_size)
        self._shard_idx = 0
        self._facies: list = []
        self._poro: list = []
        self._perm: list = []
        self._meta: list = []
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def add(self, facies, poro, perm, meta):
        self._facies.append(facies)
        self._poro.append(poro)
        self._perm.append(perm)
        self._meta.append(meta)
        if len(self._facies) >= self.shard_size:
            self._flush()

    def close(self):
        if self._facies:
            self._flush()

    def _flush(self):
        name = f"shard_r{self.rank:04d}_s{self._shard_idx:06d}"
        shard_dir = self.output_dir / name
        tmp_dir = self.output_dir / (name + ".tmp")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)

        np.save(tmp_dir / "facies.npy", np.stack(self._facies).astype(np.int8))
        np.save(tmp_dir / "poro.npy", np.stack(self._poro).astype(np.float16))
        np.save(tmp_dir / "perm.npy", np.stack(self._perm).astype(np.float16))

        # Union of keys across the shard (different layer types have disjoint
        # physics params); missing values become null in parquet.
        all_keys = sorted({k for m in self._meta for k in m.keys()})
        columns = {k: [m.get(k) for m in self._meta] for k in all_keys}
        pq.write_table(pa.Table.from_pydict(columns), tmp_dir / "params.parquet")

        if shard_dir.exists():
            raise FileExistsError(
                f"shard {shard_dir} already exists; "
                f"pick a fresh output_dir or remove the stale shard first"
            )
        os.rename(tmp_dir, shard_dir)

        self._shard_idx += 1
        self._facies.clear()
        self._poro.clear()
        self._perm.clear()
        self._meta.clear()
