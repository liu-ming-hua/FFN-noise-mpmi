from types import SimpleNamespace
from typing import Any


def adapt_single_output(output: Any) -> Any:
    """Convert one vLLM CompletionOutput into evaluator-compatible object.

    Prefer the native vLLM output structure so token-level logprobs and
    uncertainties stay aligned with the decode step semantics.
    """
    return SimpleNamespace(
        text=getattr(output, "text", ""),
        token_ids=getattr(output, "token_ids", None),
        logprobs=getattr(output, "logprobs", None),
        uncertainties=getattr(output, "uncertainties", None),
        cumulative_logprob=getattr(output, "cumulative_logprob", None),
        finish_reason=getattr(output, "finish_reason", None),
        stop_reason=getattr(output, "stop_reason", None),
    )


def adapt_request_output(request_output: Any) -> Any:
    """Convert one vLLM RequestOutput into pkl-compatible object structure."""
    outputs = getattr(request_output, "outputs", []) or []
    adapted_outputs = [adapt_single_output(o) for o in outputs]
    return SimpleNamespace(
        prompt=getattr(request_output, "prompt", ""),
        outputs=adapted_outputs,
    )
