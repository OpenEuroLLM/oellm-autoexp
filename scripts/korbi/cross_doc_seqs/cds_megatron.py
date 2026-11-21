"""Megatron's own GPTDataset / BlendedDataset objects for a cached blend, built read-only: every cache file is
reached through a temporary directory of symlinks, so nothing is ever written into the run's (shared) cache.
A configuration that does not reproduce the run's description is a failure (a regular file appears in the
symlink directory = Megatron rebuilt an index), never a silent rebuild.
"""

from __future__ import annotations

import glob
import json
import os
import tempfile

import numpy as np

from cds_data import CacheEntry, read_blend


class MegatronBlend:
    def __init__(self, entry: CacheEntry, bos: int | None, eod: int | None, vocab: int):
        self.entry = entry
        self.tmp = tempfile.mkdtemp(prefix="cds_megatron_")
        self.bos, self.eod, self.vocab = bos, eod, vocab
        self._gpt: dict = {}

    def _link(self, pattern: str) -> None:
        for f in glob.glob(os.path.join(self.entry.cache_dir, pattern)):
            dst = os.path.join(self.tmp, os.path.basename(f))
            if not os.path.exists(dst):
                os.symlink(f, dst)

    def _check_no_build(self, what: str) -> None:
        built = [f for f in os.listdir(self.tmp) if not os.path.islink(os.path.join(self.tmp, f))]
        assert not built, f"{what}: cache miss, Megatron built {built[:4]} (configuration differs)"

    def _tokenizer(self, uid):
        bos, eod, vocab = self.bos, self.eod, self.vocab

        class _Tok:  # the run's tokenizer identity, as its cache descriptions record it
            unique_identifiers = uid

        t = _Tok()
        t.bos, t.eod, t.pad, t.vocab_size = bos, eod, -1, vocab
        return t

    def config(self, d: dict, blend=None):
        from megatron.core.datasets.gpt_dataset import GPTDatasetConfig

        return GPTDatasetConfig(
            random_seed=d["random_seed"], sequence_length=d["sequence_length"], split=d["split"],
            blend=blend or ([d["dataset_path"]], None),  # without a blend the config replaces the split
            tokenizer=self._tokenizer(d["tokenizer"]), reset_position_ids=False, reset_attention_mask=False,
            eod_mask_loss=False, create_attention_mask=False, inter_document_masking=True,
            add_bos=bool(d.get("add_bos", False)), path_to_cache=self.tmp)

    def gpt(self, ds: int):
        """GPTDataset ``ds`` of the blend (train split), from the cache."""
        if ds not in self._gpt:
            from megatron.core.datasets.gpt_dataset import GPTDataset
            from megatron.core.datasets.indexed_dataset import IndexedDataset
            from megatron.core.datasets.utils import Split

            d = self.entry.desc["datasets"][ds]
            md5 = self.entry.gpt_md5(ds)
            self._link(f"{md5}-GPTDataset-train-*")
            sm = d["split_matrix"][0]
            idx = IndexedDataset(d["dataset_path"], mmap=True)
            n = len(idx.sequence_lengths)
            ids = np.arange(int(round(sm[0] * n)), int(round(sm[1] * n)), dtype=np.int32)
            g = GPTDataset(idx, d["dataset_path"], ids, d["num_samples"], Split.train, self.config(d))
            assert g.unique_description_hash == md5, f"dataset {ds}: {g.unique_description} != {json.dumps(d, indent=4)}"
            self._check_no_build(f"dataset {ds}")
            self._gpt[ds] = g
        return self._gpt[ds]

    def blended(self, data_args_path: str):
        """The run's training BlendedDataset (all GPTDatasets + the blend indices), from the cache."""
        from megatron.core.datasets.blended_dataset import BlendedDataset
        from megatron.core.datasets.utils import normalize

        weights, prefixes = read_blend(data_args_path)
        assert prefixes == self.entry.prefixes
        datasets = [self.gpt(i) for i in range(len(prefixes))]
        self._link(f"{self.entry.blend_md5}-BlendedDataset-train-*")
        cfg = self.config(self.entry.desc["datasets"][0], blend=(prefixes, weights))
        b = BlendedDataset(datasets, normalize(weights), self.entry.size, cfg)  # the builder normalizes once
        assert b.unique_description_hash == self.entry.blend_md5, "BlendedDataset description differs from the run's"
        self._check_no_build("BlendedDataset")
        return b
