from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from freetoken.core import SamplingParams
from freetoken.distributed import DistributedInfo
from freetoken.message import (
    BaseBackendMsg,
    BaseTokenizerMsg,
    CacheParkStatusMsg,
    DetokenizeMsg,
    PrefillProgressMsg,
    PromptAdmittedMsg,
    UserMsg,
)
from freetoken.scheduler import Scheduler, SchedulerConfig


class RequestAllFinished(Exception):
    pass


@dataclass
class RequestStatus:
    uid: int
    input_ids: List[int]
    output_ids: List[int]


class LLM(Scheduler):
    def __init__(self, model_path: str, dtype: torch.dtype = torch.bfloat16, **kwargs):
        config = SchedulerConfig(
            model_path=model_path,
            tp_info=DistributedInfo(0, 1),
            dtype=dtype,
            offline_mode=True,
            **kwargs,
        )
        super().__init__(config)
        self.pending_requests: List[Tuple[List[int] | str, SamplingParams, List[bytes] | None]] = []
        self.status_map: Dict[int, RequestStatus] = {}
        self.counter = 0
        self.mm_embeds_map = {}
        from freetoken.mm.processor import get_mm_processor

        self._mm_processor = get_mm_processor(model_path, config.mm)

    @torch.inference_mode()
    def encode_images(self, pixel_values: torch.Tensor, image_position_ids: torch.Tensor) -> torch.Tensor:
        """Compatibility entry point for preprocessed offline image inputs."""
        model = self.engine.model
        if not hasattr(model, "encode_images"):
            raise RuntimeError(f"{type(model).__name__} does not support legacy preprocessed image inputs")
        return model.encode_images(pixel_values.to(self.device), image_position_ids.to(self.device))

    def _tokenize_one(self, prompt: List[int] | str) -> torch.Tensor:
        if isinstance(prompt, str):
            return self.tokenizer.encode(prompt, return_tensors="pt").view(-1).to(torch.int32)
        else:
            return torch.tensor(prompt, dtype=torch.int32, device="cpu")

    def offline_receive_msg(self, blocking: bool = False) -> List[BaseBackendMsg]:
        if blocking and len(self.pending_requests) == 0:
            raise RequestAllFinished()
        results: List[BaseBackendMsg] = []
        added, sum_input_len = 0, 0
        for tokens_or_prompt, sampling_params, images in self.pending_requests:
            if sum_input_len >= self.prefill_budget:
                break
            input_ids = self._tokenize_one(tokens_or_prompt)
            msg = UserMsg(uid=0, input_ids=input_ids, sampling_params=sampling_params)
            if images:
                if self._mm_processor is None:
                    raise ValueError("image input is not supported for this model")
                r = self._mm_processor.apply(input_ids, images)
                input_ids = r.input_ids
                msg = UserMsg(
                    uid=0,
                    input_ids=input_ids,
                    sampling_params=sampling_params,
                    mm_items=r.mm_items,
                    mrope_positions=r.mrope_positions,
                    mrope_delta=r.mrope_delta,
                )
            sum_input_len += len(input_ids)
            uid, added = self.counter + added, added + 1
            msg.uid = uid
            msg.mm_embeds = getattr(self, "mm_embeds_map", {}).get(uid)
            results.append(msg)
            self.status_map[uid] = RequestStatus(
                uid=uid,
                input_ids=input_ids.tolist(),
                output_ids=[],
            )
        self.counter += added
        self.pending_requests = self.pending_requests[added:]
        return results

    def offline_send_result(self, reply: List[BaseTokenizerMsg]) -> None:
        for msg in reply:
            if isinstance(msg, (PromptAdmittedMsg, CacheParkStatusMsg, PrefillProgressMsg)):
                # These messages feed online-server accounting and status snapshots. Offline
                # generation owns its inputs and has no FrontendManager stats sink.
                continue
            assert isinstance(msg, DetokenizeMsg)
            status = self.status_map[msg.uid]
            run = msg.next_tokens
            if msg.finished and run and run[-1] in self.eos_token_ids:
                run = run[:-1]
            status.output_ids.extend(run)

    def generate(
        self,
        prompts: List[str] | List[List[int]],
        sampling_params: List[SamplingParams] | SamplingParams,
        mm_inputs: List[Dict[str, torch.Tensor] | None] | None = None,
        *,
        images: List[List[bytes] | None] | None = None,
    ) -> List[Dict[str, str | List[int]]]:
        """Offline generation; images is aligned with prompts: the raw image files of each prompt in placeholder order, or None."""
        if images is not None and mm_inputs is not None:
            raise ValueError("supply either images or mm_inputs")
        if mm_inputs is not None and len(mm_inputs) != len(prompts):
            raise ValueError("mm_inputs must align with prompts")
        self.mm_embeds_map = {}
        if mm_inputs is not None:
            for uid, mm in enumerate(mm_inputs):
                if mm is not None:
                    self.mm_embeds_map[uid] = self.encode_images(mm["pixel_values"], mm["image_position_ids"])
        self.pending_requests = []
        self.status_map = {}
        self.counter = 0
        if isinstance(sampling_params, SamplingParams):
            sampling_params = [sampling_params] * len(prompts)
        if images is None:
            images = [None] * len(prompts)
        for prompt, sp, imgs in zip(prompts, sampling_params, images, strict=True):
            self.pending_requests.append((prompt, sp, imgs))
        try:
            self.run_forever()
        except RequestAllFinished:
            pass
        results: List[Dict[str, str | List[int]]] = []
        for i in range(len(prompts)):
            status = self.status_map[i]
            output_text = self.tokenizer.decode(status.output_ids)
            results.append({"text": output_text, "token_ids": status.output_ids})
        return results
