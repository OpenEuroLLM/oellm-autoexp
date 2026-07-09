"""Monkeypatch Megatron's IndexedDataset to read through a node-local mirror.

Activated when ``OELLM_MIRROR_DIR`` is set; otherwise a no-op. Import this once
before training builds datasets (see scripts/pretrain_gpt_prefetch.py). Patches
``IndexedDataset.initialize``, which runs on both construction and unpickling, so
forked/spawned dataloader workers get the read-through reader too.
"""

from __future__ import annotations

import logging
import os

LOGGER = logging.getLogger(__name__)


def apply_windowed_shuffle() -> bool:
    """Replace Megatron's global shuffle with a locality-preserving K-lane
    transposed shuffle when OELLM_SHUFFLE_LANES is set.

    document_index -> file order; shuffle_index -> lane interleave.
    """
    lanes = os.environ.get("OELLM_SHUFFLE_LANES")
    if not lanes:
        return False
    import megatron.core.datasets.gpt_dataset as G

    if getattr(G, "_oellm_shuffle_patched", False):
        return True

    from oellm_autoexp.data_staging import windowed_shuffle as ws

    K = int(lanes)
    block = int(os.environ.get("OELLM_SHUFFLE_BLOCK", "8192"))
    seed = int(os.environ.get("OELLM_SHUFFLE_SEED", "1234"))

    _orig_doc = G._build_document_index
    _orig_shuf = G._build_shuffle_index

    def _build_document_index(documents, num_epochs, numpy_random_state, separate_final_epoch):
        # File order, tiled num_epochs times (re-randomization is per-epoch in the shuffle index).
        G._oellm_num_epochs = int(num_epochs)
        return ws.file_order_document_index(documents, int(num_epochs))

    def _build_shuffle_index(num_samples, total_size, numpy_random_state):
        # Per-epoch K-lane interleave over the full sample range (chunk-aware multi-epoch reshuffle).
        # Recover the true samples-per-epoch: when the final epoch is separated, _build_shuffle_index
        # is called with num_samples == samples sans the final (partial) epoch.
        ne = int(getattr(G, "_oellm_num_epochs", 1))
        if ne <= 1:
            epoch_size = 0
        elif num_samples == total_size:  # no separate final epoch: equal epochs
            epoch_size = total_size // ne
        else:  # separate final epoch: first (ne-1) epochs hold num_samples
            epoch_size = num_samples // (ne - 1)
        LOGGER.info(
            "OELLM windowed shuffle: K=%d block=%d epochs=%d epoch_size=%d total=%d",
            K,
            block,
            ne,
            epoch_size,
            total_size,
        )
        return ws.lane_interleave_shuffle_index(
            int(total_size), K, block, seed, epoch_size=int(epoch_size)
        )

    G._build_document_index = _build_document_index
    G._build_shuffle_index = _build_shuffle_index
    G._oellm_shuffle_patched = True
    return True


def apply_cursor() -> bool:
    """Write training's consumption progress (consumed_train_samples) to a
    node-local cursor file each iteration, so the prefetcher can pace
    staging/eviction to the GPU.

    Active when OELLM_CURSOR_FILE is set; patches
    megatron.training.training.training_log (called every iter).
    """
    cursor_file = os.environ.get("OELLM_CURSOR_FILE")
    if not cursor_file:
        return False
    import megatron.training.training as T

    if getattr(T.training_log, "_oellm_cursor", False):
        return True
    from megatron.training.global_vars import get_args

    _orig = T.training_log
    tmp = cursor_file + f".{os.getpid()}.tmp"

    def training_log(*a, **k):
        ret = _orig(*a, **k)
        try:
            with open(tmp, "w") as f:
                f.write(str(int(get_args().consumed_train_samples)))
            os.replace(tmp, cursor_file)  # atomic publish
        except Exception:  # never let cursor bookkeeping break training
            pass
        return ret

    training_log._oellm_cursor = True  # type: ignore[attr-defined]
    T.training_log = training_log
    LOGGER.info("OELLM cursor: writing consumed_train_samples to %s", cursor_file)
    return True


def apply() -> bool:
    apply_windowed_shuffle()
    apply_cursor()
    mirror_dir = os.environ.get("OELLM_MIRROR_DIR")
    if not mirror_dir:
        return False

    import megatron.core.datasets.indexed_dataset as M

    from oellm_autoexp.data_staging.mirror import LocalMirrorBinReader

    if getattr(M.IndexedDataset.initialize, "_oellm_patched", False):
        return True

    _orig = M.IndexedDataset.initialize

    def initialize(self, path_prefix, multimodal, mmap, object_storage_config=None):
        _orig(self, path_prefix, multimodal, mmap, object_storage_config)
        if object_storage_config is None:
            bin_path = M.get_bin_path(path_prefix)
            try:
                self.bin_reader = LocalMirrorBinReader(bin_path, mirror_dir)
                LOGGER.info(
                    "OELLM mirror read-through active for %s (mirror_dir=%s)", bin_path, mirror_dir
                )
            except OSError as e:  # keep the original reader on any failure
                LOGGER.warning("OELLM mirror reader unavailable (%s); using stock reader", e)

    initialize._oellm_patched = True  # type: ignore[attr-defined]
    M.IndexedDataset.initialize = initialize
    return True


apply()
