# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Quantize each eligible Linear as soon as its weights land, not after the
whole checkpoint has been read.

The classic path loads the entire BF16 model and then quantizes it in a second
pass, so peak memory holds the full-precision model plus the quantized one. With
streaming, a module eligible for online quantization is built on the meta device
(see ``LinearBase._stream_online``), materialized once its last shard arrives,
quantized immediately, and its source released -- so only the modules in flight
are ever resident at full precision.

The state lives here rather than in `loading_core` or `loader` because it spans
both: the trigger fires inside the checkpoint walk, while the coverage report
and the "already post-processed" bookkeeping are consumed after loading.
"""

import concurrent.futures
import logging
import threading

import torch
import torch.utils._python_dispatch
from torch import nn

from atom.utils import envs

logger = logging.getLogger("atom")


class _CopyCounter(torch.utils._python_dispatch.TorchDispatchMode):
    """Count the number of elements written by ``aten.copy_`` while active.

    Used by the online-quant streaming loader to detect when a layer's weights
    have fully arrived (regardless of how many partial shard/packed writes it
    took), mirroring vLLM's layerwise ``CopyCounter``.

    The dispatch mode stack is thread-local, so concurrent counters do not see
    each other's copies -- this class is safe under multi-threaded loading. What
    is *not* thread-safe is the per-module bookkeeping it feeds (see the TODO on
    forced single-threading in ``resolve_num_threads``).
    """

    def __init__(self):
        super().__init__()
        self.copied_numel = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func is torch.ops.aten.copy_.default:
            assert args[0].numel() == args[1].numel()
            self.copied_numel += args[0].numel()
        return func(*args, **kwargs)


class OnlineQuantStreamer:
    """Per-load streaming state for the online-quant path.

    One instance is created by `load_model` when streaming is both enabled and
    applicable, handed to `load_weights_into_model` to drive the trigger, and
    then read back by `load_model` for the post-load report.
    """

    @classmethod
    def maybe_create(cls, model: nn.Module, load_dummy: str | None):
        """The streamer, or None when streaming is off or has nothing to do.

        Disabled under dummy loading: no real weights arrive to trigger it.
        """
        if not envs.ATOM_ONLINE_QUANT_STREAMING or load_dummy:
            return None
        candidates = [
            (mod_name, m)
            for mod_name, m in model.named_modules()
            if getattr(m, "_stream_online", False)
        ]
        if not candidates:
            return None
        return cls(candidates)

    def __init__(self, candidates: list[tuple[str, nn.Module]]):
        self.candidates = candidates
        self.param_to_module: dict[int, nn.Module] = {}
        # id(module) handed to the streaming tail. Added when the module is
        # claimed, not when the tail finishes, so it doubles as "no more loads
        # accepted".
        self.done_module_ids: set[int] = set()
        # Loads arriving after their module was quantized.
        self.excessive_loads: list[str] = []

        self._params_dict: dict | None = None
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._futures: list = []
        # Bounds how many modules may be waiting on or inside the pool. Without
        # it the main thread would race ahead and pile up every remaining
        # module's buffered CPU sources (and, once a worker picks them up, their
        # device memory) -- exactly the peak streaming exists to avoid.
        self._slots: threading.Semaphore | None = None
        # Per-worker side stream. Sharing the default stream makes the workers
        # convoy: a blocking pageable H2D issued by one worker cannot start
        # until everything the others already queued has drained, which inflates
        # the per-module quantization cost by more than an order of magnitude
        # while buying no overlap. Each worker synchronizes its own stream before
        # the task returns, so a drained future means that module is fully
        # written.
        self._worker_stream = threading.local()

        for mod_name, m in candidates:
            # Elements (not bytes) copied into this module's params so far,
            # compared against _stream_expected_numel below to decide "complete".
            m._stream_loaded_numel = 0
            # Loader calls seen so far, held as (fn, args) until the module is
            # complete. Buffering keeps the params on meta while the module is
            # only half loaded, so nothing but the sources is resident.
            m._stream_buffer_list = []
            # params_dict keys for this module, needed to drop the pre-quant
            # Parameter objects once it is done (see _finalize).
            m._stream_param_names = [
                (f"{mod_name}.{p_name}" if mod_name else p_name, p_name)
                for p_name, p in m.named_parameters(recurse=False)
                if p is not None
            ]
            # Expected copied elements = numel of every loadable param in the
            # module (weight, plus bias if present). Computed from the meta
            # shapes; a layer that never reaches this (e.g. padded scales) falls
            # back to the post-loop pass.
            m._stream_expected_numel = sum(
                p.numel() for _, p in m.named_parameters(recurse=False) if p is not None
            )
            for _, p in m.named_parameters(recurse=False):
                if p is not None:
                    self.param_to_module[id(p)] = m

    # ── loading-loop wiring ───────────────────────────────────────────────

    def bind_params_dict(self, params_dict: dict) -> None:
        self._params_dict = params_dict

    def release_params_dict(self) -> None:
        """Drop the params_dict ref so its Parameters can be collected."""
        self._params_dict = None

    def resolve_num_threads(self, num_threads: int) -> int:
        """Force the checkpoint walk single-threaded, and say so.

        The trigger test reads and updates per-module arrival counters plus a
        shared buffer list, none of which is safe to interleave (_CopyCounter
        itself is thread-local and fine). What the walk does is cheap though --
        every copy lands in the meta sink, so it is pure bookkeeping -- and the
        expensive tail (H2D copy + dequant/requant) is offloaded to the worker
        pool instead.
        """
        if num_threads > 1:
            logger.info(
                "Online-quant streaming enabled: the checkpoint walk runs "
                "single-threaded; per-module quantization is offloaded to %d "
                "worker thread(s).",
                envs.ATOM_ONLINE_QUANT_STREAMING_THREADS,
            )
        return 1

    def start_workers(self) -> None:
        """Spin up the tail pool.

        A module's finalize (materialize on device, replay the buffered
        CPU->GPU copies, quantize, release the source) touches only that
        module's own parameters, so distinct modules are independent and can run
        while the main thread keeps looking for the next complete module.

        A fresh thread's current CUDA device defaults to 0 regardless of the
        process's, so kernels launched from a worker on rank>0 would target the
        wrong GPU. Pin it in the initializer.
        """
        num_workers = envs.ATOM_ONLINE_QUANT_STREAMING_THREADS
        if num_workers <= 0:
            return
        device = torch.cuda.current_device() if torch.cuda.is_available() else None
        worker_stream = self._worker_stream

        def _worker_init():
            if device is not None:
                torch.cuda.set_device(device)
                worker_stream.s = torch.cuda.Stream(device=device)

        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=num_workers,
            thread_name_prefix="atom-stream-quant",
            initializer=_worker_init,
        )
        self._slots = threading.Semaphore(2 * num_workers)

    def drain(self) -> None:
        """Wait for every submitted tail before anything reads the whole model.

        The post-load pass depends on it (MLA derives W_UK/W_UV from an already
        post-processed kv_b_proj), and this is also the only place a worker's
        exception can surface.
        """
        for future in concurrent.futures.as_completed(self._futures):
            future.result()
        self._futures.clear()

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    # ── trigger ───────────────────────────────────────────────────────────

    def run(self, fn, args) -> None:
        """Run one loader call, counting what it writes into a streamed module."""
        param = args[0] if args else None
        module = self.param_to_module.get(id(param)) if param is not None else None
        if module is None:
            fn(*args)
            return
        if id(module) in self.done_module_ids:
            # The module already reached _stream_expected_numel and was
            # quantized, so its params hold quantized data now. Landing here
            # means the checkpoint carries more elements for this module than
            # its params can hold, and applying the write would put
            # full-precision bytes over quantized storage. Drop it -- which is
            # what used to happen implicitly, since the write went to a
            # Parameter the module had already replaced. vLLM guards the same
            # case via `can_load()`.
            self.excessive_loads.append(
                getattr(module, "prefix", type(module).__name__)
            )
            return
        # While the param is still on meta the loader's copy_ lands in a silent
        # sink -- nothing is written, no device memory is allocated. Run it
        # anyway: that is the only way to learn how many elements it *would*
        # write, because loaders narrow/shard the source, so the incoming
        # tensor's size is not the answer. The call is buffered and replayed
        # against real storage once the module is complete, which keeps a
        # half-loaded module from pinning its full-precision storage.
        #
        # A param that already has storage (the fused-expert path materializes
        # outside _submit) is written for real here instead, and must not be
        # buffered: replaying an in-place post-load transform would apply it a
        # second time.
        deferred = param.data.is_meta
        with _CopyCounter() as counter:
            fn(*args)
        if deferred:
            module._stream_buffer_list.append((fn, args))
        # A loader call fills exactly one destination param, so it can never
        # legitimately write more than that param's size. Some loaders copy into
        # the destination more than once -- e.g. an in-place post-load transform
        # (`param.copy_(fn(param))`) layered on top of the initial copy -- which
        # would double-count, push _stream_loaded_numel past
        # _stream_expected_numel early, and quantize the module while later
        # params are still in flight (silently dropping them). Cap each call at
        # the destination size.
        module._stream_loaded_numel += min(counter.copied_numel, param.numel())
        if module._stream_loaded_numel >= module._stream_expected_numel:
            # Everything accounted for -> hand the module off (materialize,
            # apply the buffered loads for real, quantize, free the source BF16)
            # and go straight back to walking the checkpoint.
            self._submit_finalize(module)

    def materialize_fused_param(self, param: nn.Parameter) -> None:
        """Hook for the fused-expert path, which writes past `run`.

        Fused-expert loading writes straight into the param, bypassing
        _submit/run. Materialize a streaming meta buffer first so the copy has
        real storage; that module then falls back to the post-loop online-quant
        pass (its copies are not stream-counted).
        """
        module = self.param_to_module.get(id(param))
        if module is not None and param.data.is_meta:
            self._materialize_meta(param, module._load_device)

    # ── tail ──────────────────────────────────────────────────────────────

    @staticmethod
    def _materialize_meta(param: nn.Parameter, device) -> None:
        """Give a meta streaming param real, zero-initialized storage in place.

        Zeroing (not bare ``torch.empty``) matches the classic create-time
        state: methods that pad weights zero the padding region there, and the
        loader only copies the checkpoint's logical rows -- so any
        padded/unwritten elements must be zero here too, regardless of source
        format. Done via a uint8 byte view so it also works for FP4/FP8 dtypes
        that ``zero_()`` doesn't support directly.

        The swap (rather than ``param.data = buf``) is required, not stylistic:
        assigning real storage onto a meta Parameter raises "variable and tensor
        have incompatible tensor type", because meta carries a different
        dispatch key. ``swap_tensors`` exchanges the underlying TensorImpl while
        keeping the same Python object, so every reference the loader captured
        (``params_dict``, ``param_to_module``'s ``id(param)`` keys, the owning
        module's registry) stays valid. It also swaps ``__dict__``, which would
        take the stamped ``weight_loader``/``weight_loader_process`` attributes
        with it, so those are carried back over afterwards.
        """
        buf = torch.empty(tuple(param.shape), dtype=param.dtype, device=device)
        buf.view(torch.uint8).zero_()
        attrs = param.__dict__.copy()
        torch.utils.swap_tensors(
            param, nn.Parameter(buf, requires_grad=param.requires_grad)
        )
        param.__dict__.update(attrs)

    def _replay(self, module: nn.Module) -> None:
        """Give the module real storage and apply its buffered loader calls.

        Every remaining meta param is materialized, not just the buffered
        targets: post-processing reads the module as a whole, so a param the
        checkpoint never provided still needs (zeroed) storage.
        """
        for _, p in module.named_parameters(recurse=False):
            if p is not None and p.data.is_meta:
                self._materialize_meta(p, module._load_device)
        for fn, args in module._stream_buffer_list:
            fn(*args)
        module._stream_buffer_list.clear()

    def _finalize(self, module: nn.Module) -> None:
        """The tail of streaming for one module: everything after "complete".

        Runs on a worker (or inline when the pool is disabled). Only this
        module's parameters are touched, so concurrent finalizes of different
        modules do not interact.
        """
        self._replay(module)
        module.process_weights_after_loading()
        # Quantization swaps in fresh Parameter objects, but params_dict still
        # points at the originals, and while it does their full-precision
        # storage cannot be released. `del params_dict` only runs after the
        # whole load, so without something here the BF16 sources pile up
        # alongside the quantized weights and streaming peaks *higher* than the
        # classic post-load path.
        #
        # Release the storage but keep the Parameter object: param_to_module is
        # keyed on id(param), so dropping the object would let CPython recycle
        # that id for an unrelated tensor and produce false hits. Emptying
        # `.data` frees the bytes while every existing reference, and every id,
        # stays valid.
        params_dict = self._params_dict
        if params_dict is None:
            return
        for full_name, p_name in module._stream_param_names:
            stale = params_dict.get(full_name)
            if stale is not None and stale is not getattr(module, p_name, None):
                stale.data = torch.empty(0, dtype=stale.dtype, device=stale.device)

    def _submit_finalize(self, module: nn.Module) -> None:
        """Hand a completed module to the tail workers and return immediately."""
        # Mark it done *here*, on the main thread, rather than when the worker
        # finishes: from this point on the module's params are spoken for, so a
        # later load targeting it is surplus (see the drop in `run`) and must not
        # race the in-flight finalize.
        self.done_module_ids.add(id(module))
        if self._executor is None:
            self._finalize(module)
            return

        def _task():
            try:
                s = getattr(self._worker_stream, "s", None)
                if s is None:
                    self._finalize(module)
                else:
                    with torch.cuda.stream(s):
                        self._finalize(module)
                    s.synchronize()
            finally:
                self._slots.release()

        self._slots.acquire()
        self._futures.append(self._executor.submit(_task))

    # ── post-load report ──────────────────────────────────────────────────

    def replay_stragglers_and_report(self, is_rank0: bool) -> None:
        """Settle the modules that never triggered, then report coverage.

        Falling back is always correct (the post-load pass quantizes the module
        normally), but it forfeits the memory saving that is the whole point of
        streaming, so a module that silently never triggers must be visible.
        Known fallback causes: fused-expert loading writes past `run` so its
        copies are never counted, and padded weights can't reach
        _stream_expected_numel.
        """
        # A streaming module allocates *every* param on meta, and only the ones
        # the loader actually touched got materialized. Anything the checkpoint
        # never provided is still meta; left alone it would surface as a
        # confusing failure deep inside quantization or at first forward. Give
        # it real zeroed storage here so the unloaded-parameter warning from
        # loading stays the primary, readable signal for that bug.
        stranded = []
        for mod_name, m in self.candidates:
            # A module that never reached its trigger still holds its buffered
            # loads; dropping them would silently lose those weights, so replay
            # here. Genuinely missing params are the ones no loader ever
            # targeted -- that distinction has to be drawn *before* _replay,
            # which zero-fills every remaining meta param.
            targeted = {id(a[0]) for _, a in m._stream_buffer_list}
            for p_name, p in m.named_parameters(recurse=False):
                if p.data.is_meta and id(p) not in targeted:
                    stranded.append(f"{mod_name}.{p_name}")
            self._replay(m)

        fell_back = [
            n for n, m in self.candidates if id(m) not in self.done_module_ids
        ]
        if not is_rank0:
            return
        if stranded:
            logger.warning(
                "Online-quant streaming: %d parameter(s) were never loaded "
                "and stayed on the meta device; they have been zero-filled "
                "so post-processing can run, but the model is almost "
                "certainly wrong. First %d: %s",
                len(stranded),
                min(len(stranded), 20),
                stranded[:20],
            )
        if self.excessive_loads:
            unique = sorted(set(self.excessive_loads))
            logger.warning(
                "Online-quant streaming: dropped %d load(s) that arrived "
                "after their module was already quantized, across %d "
                "module(s). The checkpoint supplies more elements for these "
                "than _stream_expected_numel accounts for, so the surplus cannot "
                "be stored; verify the expected-size computation. First "
                "%d: %s",
                len(self.excessive_loads),
                len(unique),
                min(len(unique), 20),
                unique[:20],
            )
        log = logger.warning if fell_back else logger.info
        log(
            "Online-quant streaming: %d/%d eligible modules quantized during "
            "load, %d fell back to the post-load pass (no memory saving for "
            "those). First %d fallbacks: %s",
            len(self.candidates) - len(fell_back),
            len(self.candidates),
            len(fell_back),
            min(len(fell_back), 20),
            fell_back[:20],
        )
