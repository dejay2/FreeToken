"""Template compatibility for chronological Anthropic system messages.

Only the recognized Qwen ChatML guard is adapted. Other templates and custom
encoders retain the adapter's former leading-system conversion. A typed internal
signal selects this behavior independently of client template data.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Any


_QWEN_CONTENT_MACRO = re.compile(
    r"\{%[-]?\s*macro\s+render_content\(\s*content\s*,\s*do_vision_count\s*,"
    r"\s*is_system_content\s*=\s*false\s*\)\s*[-]?%\}"
)
_QWEN_SYSTEM_GUARD = re.compile(
    r"(?P<before>\{%[-]?\s*if\s+message\.role\s*==\s*(?P<role_quote>['\"])system(?P=role_quote)\s*[-]?%\}"
    r"\s*\{%[-]?\s*if\s+not\s+loop\.first\s*[-]?%\}\s*)"
    r"\{\{[-]?\s*raise_exception\(\s*(?P<error_quote>['\"])System message must be at the beginning\."
    r"(?P=error_quote)\s*\)\s*[-]?\}\}"
    r"(?P<after>\s*\{%[-]?\s*endif\s*[-]?%\})"
)


@lru_cache(maxsize=16)
def _qwen_system_template(template: str) -> str | None:
    """Relax only Qwen's known positional guard, retaining system validation.

    Fail closed on unknown structure: emitting ChatML into another model's
    template would be a protocol change. No checkpoint file or tokenizer is
    mutated; callers pass the derived template for this request only.
    """
    if (
        not _QWEN_CONTENT_MACRO.search(template)
        or "<|im_start|>" not in template
        or "<|im_end|>" not in template
        or len(list(_QWEN_SYSTEM_GUARD.finditer(template))) != 1
    ):
        return None
    return _QWEN_SYSTEM_GUARD.sub(
        lambda match: (
            match["before"]
            + "{{- '<|im_start|>system\\n' + (render_content(message.content, false, true)|trim) + '<|im_end|>\\n' }}"
            + match["after"]
        ),
        template,
    )


def prepare_system_messages(
    messages: list[dict[str, Any]],
    tokenizer: Any,
    tools: list[dict[str, Any]] | None,
    kwargs: dict[str, Any],
    preserve_system_order: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Resolve Anthropic chronology for the selected checkpoint."""
    seen_non_system = False
    has_later_system = False
    for message in messages:
        if message.get("role") == "system":
            if seen_non_system:
                has_later_system = True
                break
        else:
            seen_non_system = True
    if not preserve_system_order or not has_later_system:
        return messages, kwargs

    if getattr(tokenizer, "chat_template", None) or kwargs.get("chat_template"):
        template = tokenizer.get_chat_template(
            chat_template=kwargs.get("chat_template"), tools=tools,
        )
        adapted = _qwen_system_template(template)
        if adapted is not None:
            adapted_kwargs = dict(kwargs)
            adapted_kwargs["chat_template"] = adapted
            return messages, adapted_kwargs

    system_texts = [m.get("content") for m in messages if m.get("role") == "system"]
    others = [m for m in messages if m.get("role") != "system"]
    system = "\n\n".join(text for text in system_texts if text)
    if system:
        others.insert(0, {"role": "system", "content": system})
    return others, kwargs
