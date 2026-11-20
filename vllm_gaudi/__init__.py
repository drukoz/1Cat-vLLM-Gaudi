import os
import json
import sys


def _uses_lmcache_connector() -> bool:
    """Check if lmcache is configured as the KV connector.

    Detection is based on:
    - Environment variables (for programmatic usage), and
    - CLI args (for command-line usage via --kv-transfer-config).
    """

    def _is_lmcache_connector(connector_value: str) -> bool:
        """Return True if the given connector string represents an LMCache connector."""
        if not isinstance(connector_value, str):
            return False
        return "LMCache" in connector_value

    # 1. Check env var that may mirror --kv-transfer-config JSON.
    #    This supports programmatic workflows that configure KVTransferConfig
    #    and then expose it via environment instead of CLI.
    env_kv_config = os.getenv("VLLM_KV_TRANSFER_CONFIG")
    if env_kv_config:
        try:
            config = json.loads(env_kv_config)
            connector = config.get("kv_connector", "")
            if _is_lmcache_connector(connector):
                return True
        except (json.JSONDecodeError, TypeError):
            # Fall through to other detection mechanisms.
            pass

    # 2. Check a simple env var that may directly specify the connector name.
    env_kv_connector = os.getenv("VLLM_KV_CONNECTOR")
    if env_kv_connector and _is_lmcache_connector(env_kv_connector):
        return True

    # 3. Fallback: inspect CLI args for --kv-transfer-config as before.
    for i, arg in enumerate(sys.argv):
        if arg == "--kv-transfer-config" and i + 1 < len(sys.argv):
            try:
                config = json.loads(sys.argv[i + 1])
                connector = config.get("kv_connector", "")
                return _is_lmcache_connector(connector)
            except (json.JSONDecodeError, TypeError):
                return False
    return False


def register():
    """Register the HPU platform."""
    # Import lazily so that `import vllm_gaudi` (performed by vLLM's plugin
    # loader to obtain this `register` callable) does not pull in
    # `vllm_gaudi.platform` -> `vllm` at package-import time. Importing `vllm`
    # eagerly resolves `current_platform` (via torch_utils PIN_MEMORY), which
    # would re-enter the plugin loader while `vllm_gaudi` is only partially
    # initialized and `register` is not yet defined.
    from vllm_gaudi.platform import HpuPlatform

    HpuPlatform.set_torch_compile()
    # Monkey patch for LMCache
    # LMCache requires PT_HPU_GPU_MIGRATION=1
    # However, hooking torch.cuda.is_available() by
    # migration introduces a lot of issues in LMCache + Gaudi
    # Remove torch.cuda.is_available hook here as an alternative solution
    if _uses_lmcache_connector():
        HpuPlatform.adjust_cuda_hooks()

    # Apply HPU runtime monkey-patches (e.g. cleanup_dist_env_and_memory).
    # See vllm_gaudi/patches.py for details (GAUDISW-247825).
    from vllm_gaudi import patches as _hpu_patches

    _hpu_patches.apply()

    return "vllm_gaudi.platform.HpuPlatform"


def register_utils():
    """Register utility functions for the HPU platform."""
    import vllm_gaudi.utils  # noqa: F401

    vllm_gaudi.utils.patch_nixl_utils_for_hpu()
    # Install the in-process EngineCore reconfigure hook only when
    # multi-model mode is requested, to avoid heavy imports for all users.
    import os

    if os.environ.get("VLLM_HPU_MULTI_MODEL_CONFIG"):
        from vllm_gaudi.v1.engine.core_patch import install_engine_core_patch

        install_engine_core_patch()

    # Guard CUDA sync in upstream HF3FS mock client for non-CUDA platforms.
    from vllm_gaudi import patches as _hpu_patches

    _hpu_patches.patch_hf3fs_mock_client()


def _hpu_visible() -> bool:
    """True when ``habana_frameworks.torch`` imports and torch sees a Gaudi device."""
    try:
        import habana_frameworks.torch  # noqa: F401
        import torch

        return bool(hasattr(torch, "hpu") and torch.hpu.is_available())
    except Exception:
        return False


def _warn_gdn_core_decode_without_kernel(mode: int, loaded: bool, on_hpu: bool, errors, warn=None) -> str | None:
    """Warn once at plugin start when a nonzero VLLM_HPU_GDN_CORE_DECODE runs without an HPU kernel.

    Returns the logged message, or ``None`` when nothing was logged: mode 0, the extension loaded, or no HPU
    (the ops' CPU kernels need no extension). ``warn`` defaults to the vLLM logger's ``warning``.
    """
    if not mode or loaded or not on_hpu:
        return None
    reason = "; ".join(errors) if errors else "no error recorded"
    message = (f"GDN core decode (VLLM_HPU_GDN_CORE_DECODE={int(mode)}) has no HPU kernel: the gdn_fused_ops "
               f"extension did not load ({reason}); every GDN layer runs the torch decode path. Build it with "
               "tools/build_flashinfer_gaudi.py --gdn-fused-only, or set VLLM_HPU_GDN_CORE_DECODE=0.")
    if warn is None:
        from vllm.logger import init_logger

        warn = init_logger(__name__).warning
    warn(message)
    return message


def _warn_gdn_core_verify_without_kernel(verify_mode: int, decode_mode: int, loaded: bool, on_hpu: bool, errors,
                                         warn=None) -> str | None:
    """Warn once at plugin start when a nonzero VLLM_HPU_GDN_CORE_VERIFY runs without an HPU kernel.

    Returns the logged message, or ``None`` when nothing was logged: verify off, decode nonzero (its warning
    already reported the missing extension), the extension loaded, or no HPU (the ops' CPU kernels need no
    extension). ``warn`` defaults to the vLLM logger's ``warning``.
    """
    if not verify_mode or decode_mode or loaded or not on_hpu:
        return None
    reason = "; ".join(errors) if errors else "no error recorded"
    message = (f"GDN core verify (VLLM_HPU_GDN_CORE_VERIFY={int(verify_mode)}) has no HPU kernel: the gdn_fused_ops "
               f"extension did not load ({reason}); every GDN verification step runs the torch path. Build it with "
               "tools/build_flashinfer_gaudi.py --gdn-fused-only, or set VLLM_HPU_GDN_CORE_VERIFY=0.")
    if warn is None:
        from vllm.logger import init_logger

        warn = init_logger(__name__).warning
    warn(message)
    return message


def register_ops():
    """Register custom PluggableLayers for the HPU platform"""
    from flashinfer_gaudi import load_native_extensions

    load_native_extensions()
    from flashinfer_gaudi._native import KernelDbState, flashinfer_kernel_database_state, kernel_database_blocked_reason

    if flashinfer_kernel_database_state() is KernelDbState.BLOCKED:
        from vllm.logger import init_logger

        init_logger(__name__).warning(kernel_database_blocked_reason("libflashinfer_gaudi_kernels.so"))
    from vllm_gaudi import envs as _gaudi_envs

    decode_mode = _gaudi_envs.VLLM_HPU_GDN_CORE_DECODE
    verify_mode = _gaudi_envs.VLLM_HPU_GDN_CORE_VERIFY
    if decode_mode or verify_mode:
        # Before warmup: the loader prepends libgdn_fused_gaudi2.so to GC_KERNEL_PATH.
        from flashinfer_gaudi._native import gdn_fused_extension_errors, load_gdn_fused_extension

        loaded = load_gdn_fused_extension()
        on_hpu = _hpu_visible()
        errors = gdn_fused_extension_errors()
        _warn_gdn_core_decode_without_kernel(decode_mode, loaded, on_hpu, errors)
        _warn_gdn_core_verify_without_kernel(verify_mode, decode_mode, loaded, on_hpu, errors)
    # One line per process with what the loader decided (state, GC_KERNEL_PATH, combined database, errors):
    # the server log then shows which kernel database every worker really runs.
    from flashinfer_gaudi._native import kernel_database_status
    from vllm.logger import init_logger

    init_logger(__name__).info("FlashInfer-Gaudi kernel database: %s",
                               json.dumps(kernel_database_status(), sort_keys=True))
    import vllm_gaudi.attention.oot_mla  # noqa: F401
    """Register custom ops for the HPU platform."""
    import vllm_gaudi.v1.sample.hpu_rejection_sampler  # noqa: F401
    import vllm_gaudi.distributed.kv_transfer.kv_connector.v1.hpu_nixl_connector  # noqa: F401

    # Validate the optional native Triton/Bridge ABI before model warmup. In
    # hybrid mode a missing backend remains an explicit, counted fallback;
    # strict mode fails closed here.
    from vllm_gaudi.ops.triton_gaudi import prepare_if_enabled

    prepare_if_enabled()

    if os.getenv("VLLM_HPU_HETERO_KV_LAYOUT", "false").lower() == "true":
        import vllm_gaudi.distributed.kv_transfer.kv_connector.v1.hetero_hpu_nixl_connector  # noqa: F401
    import vllm_gaudi.v1.kv_offload.worker.cpu_hpu  # noqa: F401
    import vllm_gaudi.ops.hpu_attention  # noqa: F401
    import vllm_gaudi.ops.hpu_fused_moe  # noqa: F401
    import vllm_gaudi.ops.hpu_grouped_topk_router  # noqa: F401
    import vllm_gaudi.ops.hpu_layernorm  # noqa: F401
    import vllm_gaudi.ops.hpu_lora  # noqa: F401
    import vllm_gaudi.ops.hpu_mamba_mixer2  # noqa: F401
    import vllm_gaudi.ops.hpu_rotary_embedding  # noqa: F401
    import vllm_gaudi.ops.hpu_silu_and_mul  # noqa: F401
    import vllm_gaudi.ops.hpu_vocab_parallel_embedding  # noqa: F401
    import vllm_gaudi.ops.hpu_modelopt  # noqa: F401
    import vllm_gaudi.ops.hpu_compressed_tensors  # noqa: F401
    import vllm_gaudi.ops.hpu_fp8  # noqa: F401
    import vllm_gaudi.ops.hpu_mxfp4  # noqa: F401
    import vllm_gaudi.ops.hpu_hw_agnostic  # noqa: F401
    import vllm_gaudi.ops.hpu_gptq  # noqa: F401
    import vllm_gaudi.ops.hpu_awq  # noqa: F401
    import vllm_gaudi.ops.hpu_conv  # noqa: F401
    import vllm_gaudi.ops.hpu_dflash2  # noqa: F401
    import vllm_gaudi.ops.hpu_mm_encoder_attention  # noqa: F401
    import vllm_gaudi.ops.hpu_weights  # noqa: F401

    # Conditionally register HPURowParallelLinear only when chunking is enabled
    from vllm_gaudi.ops.hpu_row_parallel_linear import register as register_row_parallel

    register_row_parallel()

    # Register HPU LoRA layers only when row parallel chunking is active
    env_value = os.environ.get("VLLM_ROW_PARALLEL_CHUNKS", "1")
    try:
        row_parallel_chunks = int(env_value)
    except ValueError:
        row_parallel_chunks = 1
    if row_parallel_chunks > 1:
        from vllm_gaudi.lora.layers.hpu_row_parallel_linear import register_hpu_lora_layers

        register_hpu_lora_layers()


def register_models():
    import vllm_gaudi.models.utils  # noqa: F401
    import vllm_gaudi.models.interfaces  # noqa: F401
    import vllm_gaudi.models.bert  # noqa: F401
    from .models import register_model

    register_model()


def register_tool_parsers():
    """Register out-of-tree tool-call parsers for the HPU platform.

    Importing the package runs each parser module's
    ``ToolParserManager.register_module(...)`` call, so the parsers become
    selectable via ``--tool-call-parser <name>`` with no
    ``--tool-parser-plugin`` file. This runs in both the API-server and engine
    processes (general plugin), matching how vLLM loads tool parsers.
    """
    import vllm_gaudi.entrypoints.openai.tool_parsers  # noqa: F401
    from vllm_gaudi.entrypoints.openai.deepseek_v41_streaming import install

    install()
