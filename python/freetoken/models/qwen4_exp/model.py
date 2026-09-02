"""Qwen3.8-Flash-Next decoder stack with optional still-picture embeddings.

The residual state is ``R [T, hc_count*hidden]`` end to end: the embedding is repeated over the
``hc_count`` streams, every layer mixes them down to one ``[T, hidden]`` block input and injects
its output back, and the top-level mixer collapses them once before ``lm_head``. There is no
input/post layernorm and no final ``model.norm`` -- the hyper-connection norms are the only ones.

Layer contract (frozen): ``forward(R [T, hc*hidden], batch) -> R' [T, hc*hidden]`` with an
immediate combine::

    R  = R + ple(R, batch)                 # zero-based layer 1 only
    x, s = attn_hc.mix(R); y = (GDN | QSA)(x); R = attn_hc.combine(R, y, s)
    x, s = mlp_hc.mix(R);  y = MoE(x);        R = mlp_hc.combine(R, y, s)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import torch
from freetoken import diag
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, OPList, ParallelLMHead, VocabParallelEmbedding
from freetoken.models.blocks import BaseLLMModel
from freetoken.utils import nvtx_annotate

from .attention import Qwen4ExpAttention
from .hc import GatedResidual
from .moe import Qwen4ExpMoE
from .ple import PLELayer

if TYPE_CHECKING:
    from freetoken.core import Batch
    from freetoken.models.config import ModelConfig


def build_linear_mixer(config: ModelConfig, layer_id: int) -> BaseOP:
    """GDN mixer of a linear_attention layer (Qwen3.5's GDN with a configurable output gate)."""
    from .gdn import Qwen4ExpGatedDeltaNet

    g = config.linear_attention_group()
    return Qwen4ExpGatedDeltaNet(
        hidden_size=config.hidden_size,
        num_k_heads=g.num_key_heads,
        num_v_heads=g.num_value_heads,
        head_k_dim=g.key_head_dim,
        head_v_dim=g.value_head_dim,
        conv_kernel_size=g.conv_kernel_dim,
        rms_norm_eps=config.rms_norm_eps,
        layer_id=layer_id,
        output_gate=g.output_gate,
        # Qwen3.8's block-fp8 checkpoint keeps the GDN projections bf16 (only the routed
        # experts are quantized), so do not let expert_quant flip them to Fp8Block.
        expert_quant="none" if config.expert_quant == "fp8_block" else config.expert_quant,
        attn_quant=config.attn_quant,
        dense_quant=config.dense_quant,
    )


class Qwen4ExpDecoderLayer(BaseOP):
    """One decoder layer over the hyper-connection streams (see the module docstring for the flow)."""

    def __init__(self, config: ModelConfig, layer_id: int) -> None:
        self._layer_id = layer_id
        self._is_linear = config.is_linear_layer(layer_id)
        if self._is_linear:
            self.linear_attn = build_linear_mixer(config, layer_id)
        else:
            self.self_attn = Qwen4ExpAttention(config, layer_id)
        self.mlp = Qwen4ExpMoE(config, layer_id)
        self.attn_hyper_connection = GatedResidual(config)
        self.mlp_hyper_connection = GatedResidual(config)
        self.ple = (
            PLELayer(config, layer_id) if layer_id in config.qwen4_args.ple_layer_ids else None
        )

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, hidden: torch.Tensor, batch: Batch) -> torch.Tensor:
        if self.ple is not None:
            hidden = hidden + self.ple.forward(hidden, batch)
        block_input, inject = self.attn_hyper_connection.mix(hidden)
        if self._is_linear:
            block_output = self.linear_attn.forward(block_input)
        else:
            block_output = self.self_attn.forward(block_input, batch)
        hidden = self.attn_hyper_connection.combine(hidden, block_output, inject)
        block_input, inject = self.mlp_hyper_connection.mix(hidden)
        return self.mlp_hyper_connection.combine(hidden, self.mlp.forward(block_input), inject)


class Qwen4ExpModel(BaseOP):
    def __init__(self, config: ModelConfig) -> None:
        self.hc_count = config.qwen4_args.hc_count
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [Qwen4ExpDecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.hyper_connection_mixer = GatedResidual(config, use_combine=False)
        # plain tuple (not an OP child), so it never shows up in the state dict
        self._ple = tuple(layer.ple for layer in self.layers.op_list if layer.ple is not None)
        self._image_token_id = config.image_token_id

    @property
    def ple_layers(self) -> List[PLELayer]:
        """The PLE layers in decoder order -- the seam the loader attaches table backends to."""
        return list(self._ple)

    def prepare_mmap_ple_graph_capture(self, batch: Batch) -> None:
        tokens = (
            int(batch.input_ids.shape[0])
            if getattr(batch, "mtp_verify", False)
            else batch.padded_size
        )
        for ple in self._ple:
            table = ple.ple_embedding.table
            table.prepare_cuda_graph_capture(tokens * ple.ple_embedding.num_heads)

    def prepare_mmap_ple_graph_replay(self, batch: Batch) -> None:
        from .ple import build_ple_metadata

        meta = build_ple_metadata(batch, self._ple[0].args, batch.input_ids.device)
        pending = []
        for ple in self._ple:
            table = ple.ple_embedding.table
            row_ids = ple.ple_embedding.row_ids(meta)
            table.prefetch(row_ids)
            pending.append((table, row_ids))
        for table, row_ids in pending:
            table.prepare_cuda_graph_replay(row_ids)

    def reset_mmap_ple_graph(self) -> None:
        for ple in self._ple:
            ple.ple_embedding.table.reset_cuda_graph()

    def _merge_multimodal(
        self,
        input_ids: torch.Tensor,
        hidden: torch.Tensor,
        mm_embeds: torch.Tensor | None,
    ) -> torch.Tensor:
        if mm_embeds is None:
            return hidden
        if self._image_token_id is None:
            raise ValueError("picture features were supplied but the model has no picture token")
        if mm_embeds.ndim != 2 or mm_embeds.shape[1] != hidden.shape[1]:
            raise ValueError(
                "picture features must be [picture_tokens, hidden_size], got "
                f"{tuple(mm_embeds.shape)} for hidden size {hidden.shape[1]}"
            )
        mask = input_ids == self._image_token_id
        slots = int(mask.sum().item())
        if slots != mm_embeds.shape[0]:
            raise ValueError(
                f"picture-token slots ({slots}) do not match picture features "
                f"({mm_embeds.shape[0]})"
            )
        return hidden.masked_scatter(mask.unsqueeze(-1), mm_embeds.to(hidden.dtype))

    def _forward_state(
        self, input_ids: torch.Tensor, batch: Batch, *, capture_inputs: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        inputs_embeds = self.embed_tokens.forward(input_ids)
        inputs_embeds = self._merge_multimodal(
            input_ids, inputs_embeds, getattr(batch, "mm_embeds", None)
        )
        hidden = inputs_embeds.repeat(1, self.hc_count)
        captured_inputs = inputs_embeds if capture_inputs else None
        del inputs_embeds
        meta = None
        if self._ple:
            from .ple import build_ple_metadata, commit_ngram_context

            # The eager mirror of the graph path's staging gather (freetoken/diag.py; off by
            # default) -- the hash + row_ids build and the table's own prefetch.
            with diag.region("diag.ple_gather"):
                meta = build_ple_metadata(batch, self._ple[0].args, input_ids.device)
                for ple in self._ple:  # gather the PLE rows while the early layers run
                    ple.start_prefetch(batch, meta)
        for layer in self.layers.op_list:
            hidden = layer.forward(hidden, batch)
        if meta is not None:
            # single writer: the layers only read the context, so a second PLE layer's
            # prefetch sees the un-rolled window
            commit_ngram_context(meta, getattr(batch, "fla_metadata", None))
        return hidden, captured_inputs

    def forward_mtp_capture(
        self, input_ids: torch.Tensor, batch: Batch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Private eager seam returning final hidden, pre-final HC state, and merged inputs."""
        multi_stream, inputs_embeds = self._forward_state(
            input_ids, batch, capture_inputs=True
        )
        assert inputs_embeds is not None
        final_hidden = self.hyper_connection_mixer.mix(multi_stream)[0]
        return final_hidden, multi_stream, inputs_embeds

    def forward(self, input_ids: torch.Tensor, batch: Batch) -> torch.Tensor:
        multi_stream, _ = self._forward_state(input_ids, batch)
        return self.hyper_connection_mixer.mix(multi_stream)[0]


class Qwen4ExpForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig) -> None:
        from freetoken.models.config import embed_host_enabled, vision_execution_mode

        self._config = config
        self._mmap_ple = False
        self._vision_execution = vision_execution_mode() if config.is_multimodal else "gpu"
        # A tied lm_head projects against the SAME matrix every step, so the full-vocab GEMV
        # would drag all 1.27 GB over PCIe per token: host residency is only a win untied.
        self._embed_host = embed_host_enabled() and not config.tie_word_embeddings
        self.model = Qwen4ExpModel(config)
        if getattr(config, "lm_head_quant", "none") == "nvfp4":
            from freetoken.kernel.triton.nvfp4_linear import Nvfp4LMHead

            assert not config.tie_word_embeddings, "NVFP4 lm_head assumes untied embeddings"
            self.lm_head = Nvfp4LMHead(
                num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
            )
        elif getattr(config, "lm_head_quant", "none") == "int8":
            from freetoken.kernel.triton.int8_linear import Int8LMHead

            # parse_config only reaches int8 for an UNTIED head (a tied one is the embedding).
            assert not config.tie_word_embeddings, "int8 lm_head assumes untied embeddings"
            self.lm_head = Int8LMHead(
                num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
            )
        else:
            self.lm_head = ParallelLMHead(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
                tie_word_embeddings=config.tie_word_embeddings,
                tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
            )
        if config.is_multimodal:
            from .vision import Qwen4VisionModel

            self.visual = Qwen4VisionModel(config.vision_config)
        super().__init__()

    def weight_device_for_key(
        self, key: str, engine_device: torch.device
    ) -> torch.device:
        """Choose persistent storage without leaking Qwen key names into the engine."""
        if self._vision_execution == "layer-stream" and key.startswith("visual."):
            return torch.device("cpu")
        if getattr(self, "_embed_host", False) and key == "model.embed_tokens.weight":
            # Landed on the host unpinned here; ``load_host_tables`` re-homes it into exact-size
            # cudaHostAlloc storage (the caching pinned allocator would round 1.27 GB to 2 GB).
            return torch.device("cpu")
        return engine_device

    def weight_placement_report(self) -> str:
        lines = []
        if getattr(self, "_embed_host", False):
            embed = self.model.embed_tokens.weight
            lines.append(
                f"Token embedding: host-resident, bytes={embed.numel() * embed.element_size()}, "
                f"device={embed.device.type}"
            )
        if not hasattr(self, "visual"):
            return "\n".join(lines)
        tensors = self.visual.state_dict().values()
        count = 0
        nbytes = 0
        devices: set[str] = set()
        for tensor in tensors:
            count += 1
            nbytes += tensor.numel() * tensor.element_size()
            devices.add(tensor.device.type)
        lines.append(
            f"Picture weights: mode={self._vision_execution}, "
            f"backing={self.visual.weight_backing()}, tensors={count}, "
            f"bytes={nbytes}, devices={','.join(sorted(devices))}"
        )
        return "\n".join(lines)

    def prefetch_picture_weights(self) -> None:
        """Scheduler hook: start reading the picture weights when a picture is admitted.

        A no-op unless the weights are mapped. Called through ``getattr`` by the scheduler,
        so the engine never learns a Qwen-specific name -- same wiring as
        ``weight_device_for_key``.
        """
        visual = getattr(self, "visual", None)
        if visual is not None:
            visual.prefetch_weights()

    @torch.inference_mode()
    def encode_images(
        self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor
    ) -> torch.Tensor:
        if not hasattr(self, "visual"):
            raise RuntimeError("Qwen4-Exp picture weights are not loaded")
        # ``.device``, not ``.weight.device``: the embedding table may be host-resident, in
        # which case its weight sits on the CPU while the language model still runs on the GPU.
        embed_tokens = self.model.embed_tokens
        language_device = getattr(embed_tokens, "device", None) or embed_tokens.weight.device
        if self._vision_execution == "layer-stream":
            return self.visual.forward_layer_streamed(
                pixel_values,
                image_grid_thw,
                device=language_device,
            )
        return self.visual.forward(
            pixel_values.to(device=language_device),
            image_grid_thw.to(device=language_device),
        )

    def prepare_cuda_graph_capture(self, batch: Batch) -> None:
        if self._mmap_ple:
            self.model.prepare_mmap_ple_graph_capture(batch)

    def prepare_cuda_graph_replay(self, batch: Batch) -> None:
        if self._mmap_ple:
            self.model.prepare_mmap_ple_graph_replay(batch)

    def reset_cuda_graph(self) -> None:
        if self._mmap_ple:
            self.model.reset_mmap_ple_graph()

    def _load_host_embedding(self) -> int:
        """Re-home ``model.embed_tokens`` into exact-size pinned host storage.

        ``weight_device_for_key`` already landed it on the CPU, but as ordinary pageable
        memory; the UVA gather needs it pinned and device-mapped. Dummy-weight boots ignore
        ``device_for_key`` and leave the table on the GPU -- nothing to do there.
        """
        embed = self.model.embed_tokens
        if not self._embed_host or embed.weight.device.type != "cpu":
            return 0
        from freetoken.kernel.pinned import copy_to_pinned_tensor

        pinned = copy_to_pinned_tensor(embed.weight.contiguous())
        return embed.attach_host_table(pinned, torch.device("cuda", torch.cuda.current_device()))

    def adopt_weight_sources(self, engine_config) -> None:
        """Engine hook, straight after the weights load: hand the tower the mapping the
        loader built, so it can prefetch its extent.

        The same holder ``iter_weights`` used to install the views -- cached per checkpoint
        folder, so this is a lookup, not a second mapping. Nothing to do when the picture
        weights are resident: that covers ``ram`` mode, an FTW checkpoint, and a mapping the
        OS refused. Runs before ``weight_placement_report``, which reports what it decided.
        """
        visual = getattr(self, "visual", None)
        if visual is None:
            return
        from .weight import mmap_vision_weights

        source = mmap_vision_weights(engine_config.model_path)
        if source is not None:
            visual.attach_weight_source(source)

    def load_host_tables(self, engine_config) -> int:
        """Attach the PLE table (and any host-resident embedding) and return pinned bytes."""
        host_bytes = self._load_host_embedding()
        ple_layers = self.model.ple_layers
        if not ple_layers:
            return host_bytes
        from .ple import PinnedUVATable, ZeroTable, derive_ngram_hash_constants

        if getattr(engine_config, "use_dummy_weight", False):
            # Dummy fill leaves the int64 hash buffers garbage (a zero vocab size divides by
            # zero in the hash), so re-derive the real constants and read a zero table.
            for ple in ple_layers:
                args = ple.args
                mult, sizes, offsets = derive_ngram_hash_constants(
                    vocab_size=self._config.vocab_size,
                    ngram_size=args.ngram_size,
                    num_ngram_heads=args.num_ngram_heads,
                    ngram_vocab_size_base=args.ngram_vocab_size_base,
                    ple_layer_index=ple.ple_index,
                )
                emb = ple.ple_embedding
                emb.layer_multipliers.copy_(torch.tensor(mult, dtype=torch.int64))
                emb.ngram_heads_vocab_sizes.copy_(torch.tensor(sizes, dtype=torch.int64))
                emb.ngram_heads_offsets.copy_(torch.tensor(offsets, dtype=torch.int64))
                emb.attach_table(ZeroTable(offsets[-1] + sizes[-1], args.ngram_head_dim))
            return host_bytes

        ple_backend = getattr(engine_config, "ple_backend", "pinned")
        if ple_backend == "mmap":
            from .ple import MmapStagedTable
            from .weight import load_mmap_ple_table

            table = load_mmap_ple_table(engine_config.model_path, self._config.qwen4_args)
            self._ple_table = table  # Keep mappings alive.
            self._mmap_ple = True
            for ple in ple_layers:
                ple.ple_embedding.attach_table(
                    MmapStagedTable(table.storage, float(table.weight_scale))
                )
            return host_bytes

        if engine_config.ple_backend == "disk":
            from freetoken.utils import download_hf_weight

            from .ple_disk import DiskRowTable, resolve_row_source

            folder = download_hf_weight(engine_config.model_path)
            # one WAIT node per captured graph: the flag protocol supports a single consume
            assert len(ple_layers) == 1, "disk PLE backend expects exactly one PLE layer"
            emb, args = ple_layers[0].ple_embedding, ple_layers[0].args
            # hash with the state-dict-loaded constants, the same source the pinned path reads
            constants = {
                "num_ngram_heads": args.num_ngram_heads,
                "layer_multipliers": emb.layer_multipliers.tolist(),
                "per_head_vocab_sizes": emb.ngram_heads_vocab_sizes.tolist(),
                "per_head_offsets": emb.ngram_heads_offsets.tolist(),
                "eos_token_id": args.ngram_boundary_token_id,
            }
            disk_table = DiskRowTable(
                resolve_row_source(folder),
                constants,
                max_graph_rows=max(256, engine_config.cuda_graph_max_bs or 0),
                max_extend_tokens=engine_config.max_extend_tokens,
            )
            self._ple_table = disk_table
            for ple in ple_layers:
                ple.ple_embedding.attach_table(disk_table)
            # engine enters this around every dispatch; the graph itself never waits on the disk
            self.forward_host_ctx = disk_table.forward_host_ctx
            return 0

        if ple_backend != "pinned":
            raise ValueError(f"unsupported PLE backend {ple_backend!r}")

        from .weight import load_ple_table

        table = load_ple_table(engine_config.model_path, self._config.qwen4_args)
        self._ple_table = table  # owns the pinned HostBank; keep it alive
        for ple in ple_layers:
            ple.ple_embedding.attach_table(
                PinnedUVATable(table.bank.tensor, float(table.weight_scale))
            )
        return host_bytes + table.bank.nbytes

    def forward_mtp_capture(
        self, *, all_row_logits: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return private in-process capture data; ordinary ``forward`` remains unchanged."""
        batch = get_global_ctx().batch
        final_hidden, multi_stream, inputs_embeds = self.model.forward_mtp_capture(
            batch.input_ids, batch
        )
        logits = (
            self.lm_head.forward_all(final_hidden)
            if all_row_logits
            else self.lm_head.forward(final_hidden)
        )
        return logits, multi_stream, inputs_embeds

    def forward(self) -> torch.Tensor:
        batch = get_global_ctx().batch
        return self.lm_head.forward(self.model.forward(batch.input_ids, batch))


__all__ = ["Qwen4ExpDecoderLayer", "Qwen4ExpForCausalLM", "Qwen4ExpModel", "build_linear_mixer"]
