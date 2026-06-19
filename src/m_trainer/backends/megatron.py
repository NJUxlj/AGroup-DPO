"""Megatron-LM 后端适配器 (FR-07)

基于 NVIDIA Megatron-LM 的 Tensor Parallelism（张量并行）实现模型并行训练。
当 tensor_model_parallel_size >= 2 时，自动将 HuggingFace 模型中的 Linear / Embedding
层替换为 Megatron 的 ColumnParallelLinear / RowParallelLinear / VocabParallelEmbedding，
并将权重按 TP rank 分片。

支持架构：
  - GPT2（含 GPT2LMHeadModel / GPT2Model）
  - Qwen2.5（Qwen2ForCausalLM / Qwen2Model）
  - Qwen3（与 Qwen2.5 共享解码器层架构）

参考文档：
  - Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism
  - https://docs.nvidia.com/megatron-core/developer-guide/
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

from utils.logger import CustomLogger

from .base import DistributedBackend, TrainerConfig

log = CustomLogger.get_logger(__name__)

# ---------------------------------------------------------------------------
# 模型架构检测
# ---------------------------------------------------------------------------


def detect_model_type(model: nn.Module) -> str:
    """自动检测 HuggingFace 模型架构类型。

    按优先级依次检查 model.config.model_type → config.architectures → class name。

    Args:
        model: HuggingFace PreTrainedModel 实例

    Returns:
        'gpt2' | 'qwen2' | 若无法识别则返回 'unknown'
    """
    cls_name = model.__class__.__name__
    config = getattr(model, 'config', None)

    model_type = ''
    architectures: list[str] = []
    if config is not None:
        model_type = getattr(config, 'model_type', '') or ''
        architectures = list(getattr(config, 'architectures', []) or [])

    # Qwen2 / Qwen3 —— 共享相同解码器层架构
    if 'qwen2' in model_type.lower() or 'qwen3' in model_type.lower():
        return 'qwen2'
    for arch in architectures:
        if 'Qwen2' in arch or 'Qwen3' in arch:
            return 'qwen2'

    # GPT2
    if 'gpt2' in model_type.lower():
        return 'gpt2'
    for arch in architectures:
        if 'GPT2' in arch:
            return 'gpt2'

    # class name 兜底
    cls_lower = cls_name.lower()
    if 'gpt2' in cls_lower:
        return 'gpt2'
    if 'qwen2' in cls_lower or 'qwen3' in cls_lower:
        return 'qwen2'

    return 'unknown'


# ---------------------------------------------------------------------------
# Tensor Parallelism 辅助工具
# ---------------------------------------------------------------------------


def _is_megatron_core_available() -> bool:
    """检查 megatron-core 是否已安装。"""
    try:
        import megatron.core  # noqa: F401
        return True
    except ImportError:
        return False


def _get_tp_rank_and_size() -> tuple[int, int]:
    """获取当前进程的 TP rank 和 world size。

    若 megatron.core 未安装或分布式环境未初始化，返回 (0, 1)。
    """
    try:
        from megatron.core.parallel_state import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )
        return get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size()
    except (ImportError, AssertionError):
        return 0, 1


# ---------------------------------------------------------------------------
# TP Wrapper 层 —— 封装 megatron.core 层的返回格式差异
# ---------------------------------------------------------------------------


class _ColumnParallelWrapper(nn.Module):
    """ColumnParallelLinear 包装器。

    Megatron 的 ColumnParallelLinear.forward() 返回 (output, bias) 元组，
    本包装器将其转为标准 tensor，保证与原始 Linear 层接口兼容。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        tp_config: Any,
        init_method: Any,
        gather_output: bool = False,
    ):
        super().__init__()
        from megatron.core.tensor_parallel.layers import ColumnParallelLinear

        self._tp_linear = ColumnParallelLinear(
            input_size=in_features,
            output_size=out_features,
            bias=bias,
            config=tp_config,
            init_method=init_method,
            gather_output=gather_output,
        )
        # 外挂 bias（当 gather_output=False 时 bias 不会 all-gather）
        self._external_bias: Optional[nn.Parameter] = None

    def load_weights_from_linear(self, src: nn.Linear) -> None:
        """从标准 nn.Linear 加载权重并按 TP rank 分片。

        ColumnParallel 按 output 维度（dim=0）切分权重。
        """
        tp_rank, tp_size = _get_tp_rank_and_size()
        weight = src.weight.data.detach().clone()
        chunk_size = weight.size(0) // tp_size
        assert weight.size(0) % tp_size == 0, (
            f"output_size {weight.size(0)} must be divisible by tp_size {tp_size}"
        )
        self._tp_linear.weight.data.copy_(
            weight[tp_rank * chunk_size:(tp_rank + 1) * chunk_size]
        )
        # bias: 每个 rank 各自持有分片
        if src.bias is not None:
            bias_chunk = src.bias.data.detach().clone().chunk(tp_size)[tp_rank]
            if self._tp_linear.bias is not None:
                self._tp_linear.bias.data.copy_(bias_chunk)
            else:
                self._external_bias = nn.Parameter(bias_chunk)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, bias = self._tp_linear(x)
        if bias is not None:
            output = output + bias
        elif self._external_bias is not None:
            output = output + self._external_bias
        return output


class _RowParallelWrapper(nn.Module):
    """RowParallelLinear 包装器。

    RowParallel 按 input 维度（dim=1）切分权重，forward 中自动 all-reduce。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        tp_config: Any,
        init_method: Any,
        input_is_parallel: bool = False,
    ):
        super().__init__()
        from megatron.core.tensor_parallel.layers import RowParallelLinear

        self._tp_linear = RowParallelLinear(
            input_size=in_features,
            output_size=out_features,
            bias=bias,
            config=tp_config,
            init_method=init_method,
            input_is_parallel=input_is_parallel,
            skip_bias_add=True,  # bias 在外部加法，保持接口与标准 Linear 一致
        )
        self._external_bias: Optional[nn.Parameter] = None

    def load_weights_from_linear(self, src: nn.Linear) -> None:
        """从标准 nn.Linear 加载权重并按 TP rank 分片。

        RowParallel 按 input 维度（dim=1 / weight 的第二维）切分权重。
        """
        tp_rank, tp_size = _get_tp_rank_and_size()
        weight = src.weight.data.detach().clone()
        chunk_size = weight.size(1) // tp_size
        assert weight.size(1) % tp_size == 0, (
            f"input_size {weight.size(1)} must be divisible by tp_size {tp_size}"
        )
        self._tp_linear.weight.data.copy_(
            weight[:, tp_rank * chunk_size:(tp_rank + 1) * chunk_size]
        )
        # bias 不分片（RowParallel 的 bias 每个 rank 都持有完整副本）
        if src.bias is not None:
            if self._tp_linear.bias is not None:
                self._tp_linear.bias.data.copy_(src.bias.data.detach().clone())
            else:
                self._external_bias = nn.Parameter(src.bias.data.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, bias = self._tp_linear(x)
        if bias is not None:
            output = output + bias
        elif self._external_bias is not None:
            output = output + self._external_bias
        return output


class _VocabParallelEmbeddingWrapper(nn.Module):
    """VocabParallelEmbedding 包装器。

    按 vocab 维度（dim=0）切分嵌入表。
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        tp_config: Any,
        init_method: Any,
    ):
        super().__init__()
        from megatron.core.tensor_parallel.layers import VocabParallelEmbedding

        self._tp_embed = VocabParallelEmbedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            config=tp_config,
            init_method=init_method,
        )

    def load_weights_from_embedding(self, src: nn.Embedding) -> None:
        """从标准 nn.Embedding 加载权重并按 TP rank 分片。"""
        tp_rank, tp_size = _get_tp_rank_and_size()
        weight = src.weight.data.detach().clone()
        chunk_size = weight.size(0) // tp_size
        assert weight.size(0) % tp_size == 0, (
            f"vocab_size {weight.size(0)} must be divisible by tp_size {tp_size}"
        )
        self._tp_embed.weight.data.copy_(
            weight[tp_rank * chunk_size:(tp_rank + 1) * chunk_size]
        )

    @property
    def weight(self) -> nn.Parameter:
        return self._tp_embed.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._tp_embed(x)


# ---------------------------------------------------------------------------
# 模型转换 —— GPT2
# ---------------------------------------------------------------------------


def _convert_gpt2_to_tp(
    model: nn.Module,
    tp_config: Any,
    init_method: Any,
) -> nn.Module:
    """将 GPT2 模型中的 Linear/Embedding 层替换为 TP 版本。

    GPT2 转换策略：
      - wte (Embedding)      → VocabParallelEmbedding
      - c_attn (QKV fused)   → ColumnParallelLinear（按 output 分片）
      - c_proj (attn output) → RowParallelLinear
      - c_fc (MLP up)        → ColumnParallelLinear
      - c_proj (MLP down)    → RowParallelLinear
      - lm_head              → ColumnParallelLinear（如与 wte 共享则跳过）
    """
    tp_rank, tp_size = _get_tp_rank_and_size()
    if tp_size <= 1:
        return model

    transformer = getattr(model, 'transformer', model)

    # --- wte: 词嵌入并行 ---
    if hasattr(transformer, 'wte') and isinstance(transformer.wte, nn.Embedding):
        old_wte = transformer.wte
        new_wte = _VocabParallelEmbeddingWrapper(
            num_embeddings=old_wte.num_embeddings,
            embedding_dim=old_wte.embedding_dim,
            tp_config=tp_config,
            init_method=init_method,
        )
        new_wte.load_weights_from_embedding(old_wte)
        transformer.wte = new_wte
        log.info(
            "GPT2 wte → VocabParallelEmbedding (rank=%s/%s)", tp_rank, tp_size
        )

    # --- lm_head ---
    if hasattr(model, 'lm_head') and isinstance(model.lm_head, nn.Linear):
        old_head = model.lm_head
        new_head = _ColumnParallelWrapper(
            in_features=old_head.in_features,
            out_features=old_head.out_features,
            bias=old_head.bias is not None,
            tp_config=tp_config,
            init_method=init_method,
            gather_output=True,
        )
        new_head.load_weights_from_linear(old_head)
        model.lm_head = new_head
        log.info(
            "GPT2 lm_head → ColumnParallelLinear (rank=%s/%s)", tp_rank, tp_size
        )

    # --- transformer blocks ---
    blocks = getattr(transformer, 'h', None)
    if blocks is None:
        return model

    for i, block in enumerate(blocks):
        # -- attention --
        if hasattr(block, 'attn'):
            attn = block.attn
            # c_attn: QKV 联合投影 → ColumnParallel
            if hasattr(attn, 'c_attn') and isinstance(attn.c_attn, nn.Linear):
                old = attn.c_attn
                new = _ColumnParallelWrapper(
                    in_features=old.in_features,
                    out_features=old.out_features,
                    bias=old.bias is not None,
                    tp_config=tp_config,
                    init_method=init_method,
                    gather_output=False,
                )
                new.load_weights_from_linear(old)
                attn.c_attn = new

            # c_proj: attention output → RowParallel
            if hasattr(attn, 'c_proj') and isinstance(attn.c_proj, nn.Linear):
                old = attn.c_proj
                new = _RowParallelWrapper(
                    in_features=old.in_features,
                    out_features=old.out_features,
                    bias=old.bias is not None,
                    tp_config=tp_config,
                    init_method=init_method,
                    input_is_parallel=True,
                )
                new.load_weights_from_linear(old)
                attn.c_proj = new

        # -- MLP --
        if hasattr(block, 'mlp'):
            mlp = block.mlp
            # c_fc → ColumnParallel
            if hasattr(mlp, 'c_fc') and isinstance(mlp.c_fc, nn.Linear):
                old = mlp.c_fc
                new = _ColumnParallelWrapper(
                    in_features=old.in_features,
                    out_features=old.out_features,
                    bias=old.bias is not None,
                    tp_config=tp_config,
                    init_method=init_method,
                    gather_output=False,
                )
                new.load_weights_from_linear(old)
                mlp.c_fc = new

            # c_proj → RowParallel
            if hasattr(mlp, 'c_proj') and isinstance(mlp.c_proj, nn.Linear):
                old = mlp.c_proj
                new = _RowParallelWrapper(
                    in_features=old.in_features,
                    out_features=old.out_features,
                    bias=old.bias is not None,
                    tp_config=tp_config,
                    init_method=init_method,
                    input_is_parallel=True,
                )
                new.load_weights_from_linear(old)
                mlp.c_proj = new

        log.debug("GPT2 block %s converted to TP", i)

    log.info(
        "GPT2 model converted to TP: rank=%s/%s, blocks=%s",
        tp_rank, tp_size, len(blocks),
    )
    return model


# ---------------------------------------------------------------------------
# 模型转换 —— Qwen2 / Qwen3
# ---------------------------------------------------------------------------


def _convert_qwen2_to_tp(
    model: nn.Module,
    tp_config: Any,
    init_method: Any,
) -> nn.Module:
    """将 Qwen2.5/Qwen3 模型中的 Linear/Embedding 层替换为 TP 版本。

    Qwen2 转换策略：
      - embed_tokens          → VocabParallelEmbedding
      - q_proj / k_proj / v_proj → ColumnParallelLinear
      - o_proj                → RowParallelLinear
      - gate_proj / up_proj   → ColumnParallelLinear
      - down_proj             → RowParallelLinear
      - lm_head               → ColumnParallelLinear（gather_output=True）
    """
    tp_rank, tp_size = _get_tp_rank_and_size()
    if tp_size <= 1:
        return model

    base_model = getattr(model, 'model', model)

    # --- embed_tokens ---
    if hasattr(base_model, 'embed_tokens') and isinstance(
        base_model.embed_tokens, nn.Embedding
    ):
        old_embed = base_model.embed_tokens
        new_embed = _VocabParallelEmbeddingWrapper(
            num_embeddings=old_embed.num_embeddings,
            embedding_dim=old_embed.embedding_dim,
            tp_config=tp_config,
            init_method=init_method,
        )
        new_embed.load_weights_from_embedding(old_embed)
        base_model.embed_tokens = new_embed
        log.info(
            "Qwen2 embed_tokens → VocabParallelEmbedding (rank=%s/%s)",
            tp_rank, tp_size,
        )

    # --- lm_head ---
    if hasattr(model, 'lm_head') and isinstance(model.lm_head, nn.Linear):
        old_head = model.lm_head
        new_head = _ColumnParallelWrapper(
            in_features=old_head.in_features,
            out_features=old_head.out_features,
            bias=old_head.bias is not None,
            tp_config=tp_config,
            init_method=init_method,
            gather_output=True,
        )
        new_head.load_weights_from_linear(old_head)
        model.lm_head = new_head
        log.info(
            "Qwen2 lm_head → ColumnParallelLinear (rank=%s/%s)", tp_rank, tp_size
        )

    # --- decoder layers ---
    layers = getattr(base_model, 'layers', None)
    if layers is None:
        return model

    for i, layer in enumerate(layers):
        attn = getattr(layer, 'self_attn', None)
        if attn is not None:
            # q_proj / k_proj / v_proj → ColumnParallel
            for proj_name in ('q_proj', 'k_proj', 'v_proj'):
                old_proj = getattr(attn, proj_name, None)
                if old_proj is not None and isinstance(old_proj, nn.Linear):
                    new_proj = _ColumnParallelWrapper(
                        in_features=old_proj.in_features,
                        out_features=old_proj.out_features,
                        bias=old_proj.bias is not None,
                        tp_config=tp_config,
                        init_method=init_method,
                        gather_output=False,
                    )
                    new_proj.load_weights_from_linear(old_proj)
                    setattr(attn, proj_name, new_proj)

            # o_proj → RowParallel
            old_o = getattr(attn, 'o_proj', None)
            if old_o is not None and isinstance(old_o, nn.Linear):
                new_o = _RowParallelWrapper(
                    in_features=old_o.in_features,
                    out_features=old_o.out_features,
                    bias=old_o.bias is not None,
                    tp_config=tp_config,
                    init_method=init_method,
                    input_is_parallel=True,
                )
                new_o.load_weights_from_linear(old_o)
                attn.o_proj = new_o

        # MLP
        mlp = getattr(layer, 'mlp', None)
        if mlp is not None:
            # gate_proj / up_proj → ColumnParallel
            for proj_name in ('gate_proj', 'up_proj'):
                old_proj = getattr(mlp, proj_name, None)
                if old_proj is not None and isinstance(old_proj, nn.Linear):
                    new_proj = _ColumnParallelWrapper(
                        in_features=old_proj.in_features,
                        out_features=old_proj.out_features,
                        bias=old_proj.bias is not None,
                        tp_config=tp_config,
                        init_method=init_method,
                        gather_output=False,
                    )
                    new_proj.load_weights_from_linear(old_proj)
                    setattr(mlp, proj_name, new_proj)

            # down_proj → RowParallel
            old_down = getattr(mlp, 'down_proj', None)
            if old_down is not None and isinstance(old_down, nn.Linear):
                new_down = _RowParallelWrapper(
                    in_features=old_down.in_features,
                    out_features=old_down.out_features,
                    bias=old_down.bias is not None,
                    tp_config=tp_config,
                    init_method=init_method,
                    input_is_parallel=True,
                )
                new_down.load_weights_from_linear(old_down)
                mlp.down_proj = new_down

        log.debug("Qwen2 layer %s converted to TP", i)

    log.info(
        "Qwen2 model converted to TP: rank=%s/%s, layers=%s",
        tp_rank, tp_size, len(layers),
    )
    return model


# ---------------------------------------------------------------------------
# MegatronBackend
# ---------------------------------------------------------------------------


class MegatronBackend(DistributedBackend):
    """Megatron-LM Tensor Parallelism 后端。

    使用 NVIDIA Megatron-LM 的 megatron.core.tensor_parallel 模块
    将 HuggingFace 模型中的 Attention / MLP 线性层替换为 TP 版本，
    实现权重按 GPU 分片。

    适用场景：
      - 2 卡：tensor_model_parallel_size=2，每卡持有 1/2 层权重
      - TP=1：退化为标准 DDP 行为（无层内分片）

    模型支持：GPT2 / Qwen2.5 / Qwen3

    Usage:
        backend = MegatronBackend()
        model, optimizer = backend.init(model, optimizer=None, config=cfg)
        for batch in dataloader:
            loss = model(**batch).loss
            backend.backward(loss)
            backend.step()
    """

    def __init__(self) -> None:
        self._model: Optional[nn.Module] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._config: Optional[TrainerConfig] = None
        self._tp_size: int = 1
        self._megatron_initialized: bool = False

    # ---- DistributedBackend 接口实现 ----

    def init(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        config: TrainerConfig,
    ) -> tuple[nn.Module, Any]:
        """Megatron 分布式初始化。

        步骤：
          1. 检测模型架构（GPT2 / Qwen2）
          2. 读取 megatron_config 中的 TP/PP 参数
          3. 初始化 Megatron 并行通信组
          4. 将模型中的 Linear/Embedding 层替换为 TP 版本
          5. 创建优化器（若外部未传入）

        Args:
            model: HuggingFace 模型（未包装）
            optimizer: 外部优化器或 None
            config: 训练器配置

        Returns:
            (tp_model, optimizer): TP 转换后的模型与优化器

        Raises:
            ImportError: 若 megatron-core 未安装
            ValueError: 若模型架构不受支持
        """
        self._config = config

        # 1. 检测模型类型
        model_type = detect_model_type(model)
        if model_type == 'unknown':
            raise ValueError(
                f"Unsupported model type: {model.__class__.__name__}. "
                f"MegatronBackend supports GPT2, Qwen2.5, and Qwen3 models. "
                f"Set model.config.model_type or model.config.architectures accordingly."
            )
        log.info("Detected model type: %s", model_type)

        # 2. 读取 Megatron 配置
        mg_config = config.megatron_config
        self._tp_size = mg_config.get('tensor_model_parallel_size', 1)
        pp_size = mg_config.get('pipeline_model_parallel_size', 1)

        # 若 world_size > 1 且未显式设置 TP，自动推断
        if self._tp_size <= 1 and config.world_size > 1:
            self._tp_size = config.world_size
            log.info(
                "Auto-inferred TP size from world_size: %s", self._tp_size
            )

        # 3. 初始化 Megatron 并行环境
        if self._tp_size > 1:
            self._init_megatron_parallel(config)

        # 4. 模型 TP 转换
        tp_model = self._apply_tensor_parallelism(model, model_type, config)

        # 5. 创建优化器
        if optimizer is None:
            optimizer = self._create_optimizer(tp_model, config)

        self._model = tp_model
        self._optimizer = optimizer

        log.info(
            "Megatron backend initialized: tp_size=%s, model_type=%s, device=%s",
            self._tp_size,
            model_type,
            next(tp_model.parameters()).device,
        )
        return tp_model, optimizer

    def wrap_model(self, model: nn.Module) -> nn.Module:
        """对模型应用 TP 转换。

        若已通过 init() 转换过，直接返回 _model。
        """
        if self._model is not None:
            return self._model
        if self._tp_size > 1 and self._config is not None:
            model_type = detect_model_type(model)
            return self._apply_tensor_parallelism(model, model_type, self._config)
        return model

    def wrap_optimizer(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.Optimizer:
        """Megatron 下优化器无需额外包装，直接返回。"""
        return optimizer

    def prepare_dataloader(self, dataloader: DataLoader) -> DataLoader:
        """注入 DistributedSampler（若尚未注入）。

        Megatron TP 不改变数据并行维度，因此 sampler 与标准 DDP 一致。
        """
        if not isinstance(dataloader.sampler, DistributedSampler):
            if torch.distributed.is_initialized():
                sampler = DistributedSampler(dataloader.dataset)  # type: ignore[arg-type]
                dataloader = DataLoader(
                    dataloader.dataset,
                    batch_size=dataloader.batch_size,
                    sampler=sampler,
                    num_workers=dataloader.num_workers,
                    pin_memory=dataloader.pin_memory,
                    drop_last=dataloader.drop_last,
                )
        return dataloader

    def barrier(self) -> None:
        """跨所有并行组的分布式同步屏障。"""
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def state_dict(self) -> dict[str, Any]:
        """获取模型与优化器状态字典。

        Returns:
            {'model': model_state, 'optimizer': opt_state}

        Note:
            TP 模型各 rank 持有不同参数分片，state_dict 仅在当前 rank 有效。
            跨 TP size 的 checkpoint 恢复需在相同 TP 配置下进行。
        """
        result: dict[str, Any] = {}
        if self._model is not None:
            result['model'] = self._model.state_dict()
        if self._optimizer is not None:
            result['optimizer'] = self._optimizer.state_dict()
        result['tp_size'] = self._tp_size
        return result

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """从状态字典恢复模型与优化器。

        会校验 tp_size 一致性，防止不同 TP 配置间误加载。
        """
        saved_tp = state.get('tp_size', 1)
        if saved_tp != self._tp_size:
            log.warning(
                "Checkpoint tp_size=%s differs from current tp_size=%s. "
                "Cross-TP checkpoint loading requires merge_and_unload first.",
                saved_tp,
                self._tp_size,
            )
        if self._model is not None and 'model' in state:
            self._model.load_state_dict(state['model'], strict=False)
        if self._optimizer is not None and 'optimizer' in state:
            self._optimizer.load_state_dict(state['optimizer'])

    def backward(self, loss: torch.Tensor) -> None:
        """反向传播。

        Megatron TP 层的反向传播自动处理 all-reduce 通信，
        因此直接调用 loss.backward() 即可。
        """
        loss.backward()

    def step(self) -> None:
        """执行一步优化器更新。"""
        if self._optimizer is not None:
            self._optimizer.step()

    def zero_grad(self) -> None:
        """清零梯度。"""
        if self._optimizer is not None:
            self._optimizer.zero_grad()

    # ---- 内部方法 ----

    def _init_megatron_parallel(self, config: TrainerConfig) -> None:
        """初始化 Megatron 分布式并行环境。

        包括 torch.distributed init（若未初始化）和
        megatron.core.parallel_state 的模型并行通信组初始化。
        """
        if not _is_megatron_core_available():
            raise ImportError(
                "megatron-core is not installed. "
                "Install with: pip install megatron-core"
            )

        # 确保 torch.distributed 已初始化
        if not torch.distributed.is_initialized():
            log.info("Initializing torch.distributed with NCCL backend")
            torch.distributed.init_process_group(backend='nccl')

        try:
            from megatron.core.parallel_state import (
                initialize_model_parallel,
                destroy_model_parallel,
            )
        except ImportError as e:
            raise ImportError(
                "Failed to import megatron.core.parallel_state. "
                f"Ensure megatron-core is installed. Original error: {e}"
            ) from e

        # 清理旧并行组（防止重复初始化）
        try:
            destroy_model_parallel()
        except Exception:
            pass

        pp_size = config.megatron_config.get('pipeline_model_parallel_size', 1)
        initialize_model_parallel(
            tensor_model_parallel_size=self._tp_size,
            pipeline_model_parallel_size=pp_size,
        )
        self._megatron_initialized = True

        log.info(
            "Megatron model parallel initialized: tp=%s, pp=%s",
            self._tp_size,
            pp_size,
        )

    def _apply_tensor_parallelism(
        self,
        model: nn.Module,
        model_type: str,
        config: TrainerConfig,
    ) -> nn.Module:
        """根据模型类型应用 Tensor Parallelism 转换。"""
        if self._tp_size <= 1:
            log.info(
                "TP size=%s — skipping tensor parallelism (DDP-only mode)",
                self._tp_size,
            )
            return model

        if not _is_megatron_core_available():
            log.warning(
                "megatron-core not installed; falling back to non-TP model. "
                "Install with: pip install megatron-core"
            )
            return model

        from megatron.core import ModelParallelConfig
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

        tp_config = ModelParallelConfig()
        model_parallel_cuda_manual_seed(config.seed)

        init_method = self._build_init_method(model)

        if model_type == 'gpt2':
            return _convert_gpt2_to_tp(model, tp_config, init_method)
        elif model_type == 'qwen2':
            return _convert_qwen2_to_tp(model, tp_config, init_method)
        else:
            log.warning(
                "Unknown model type '%s', TP conversion skipped", model_type
            )
            return model

    @staticmethod
    def _build_init_method(model: nn.Module) -> Any:
        """构造与原始模型权重 std 一致的初始化方法。

        从模型中采样一个 Linear 层，计算其权重的标准差，
        用于 Megatron TP 层的权重初始化。
        """
        sigma = 0.02  # GPT/Qwen 系列默认
        for module in model.modules():
            if isinstance(module, nn.Linear) and module.weight.numel() > 0:
                sigma = module.weight.data.std().item()
                if sigma > 0:
                    break

        def _init(tensor: torch.Tensor) -> None:
            nn.init.normal_(tensor, mean=0.0, std=sigma)

        return _init

    @staticmethod
    def _create_optimizer(
        model: nn.Module,
        config: TrainerConfig,
    ) -> torch.optim.Optimizer:
        """创建 AdamW 优化器。

        Megatron 后端在 TP=1 时使用标准 PyTorch 优化器；
        TP>1 时 Megatron 层的参数与标准参数无异，可直接使用。
        """
        lr = config.learning_rate
        betas = config.megatron_config.get('adam_betas', (0.9, 0.999))
        weight_decay = config.megatron_config.get('weight_decay', 0.0)
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )
