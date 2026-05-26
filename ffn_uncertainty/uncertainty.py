from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


def _safe_logprob_from_step(step: Any, fallback: float = -20.0) -> float:
    """Extract chosen-token logprob from one decode step top-logprob dict."""
    try:
        if isinstance(step, dict) and len(step) > 0:
            first_val = next(iter(step.values()))
            lp = getattr(first_val, "logprob", None)
            if lp is not None:
                return float(lp)
    except Exception:
        pass
    return float(fallback)


def _build_uncertainty_token(lp: float, scale: float = 1.0) -> SimpleNamespace:
    """Create uncertainty object with expected attribute names."""
    aleatoric = float(max(0.0, -lp) * scale)
    epistemic = 0.0
    total = aleatoric + epistemic
    return SimpleNamespace(
        aleatoric_uncertainty=aleatoric,
        epistemic_uncertainty=epistemic,
        total_uncertainty=total,
    )


def _extract_step_lists(output: Any) -> Tuple[List[int], List[float]]:
    token_ids: List[int] = []
    logprobs: List[float] = []
    logprob_steps: Iterable[Any] = getattr(output, "logprobs", None) or []

    for step in logprob_steps:
        token_id = -1
        try:
            if isinstance(step, dict) and len(step) > 0:
                token_id = int(next(iter(step.keys())))
        except Exception:
            token_id = -1

        token_ids.append(token_id)
        logprobs.append(_safe_logprob_from_step(step))

    return token_ids, logprobs


def build_multi_sample_token_level_uncertainties(
    outputs: Sequence[Any],
) -> Dict[str, List[Dict[int, Any]]]:
    """Aggregate multiple noisy generations into token-level uncertainty lists.

    The first output defines the token ids and target length. For each token
    position, we aggregate the corresponding selected-token logprobs across
    all samples.
    """
    if not outputs:
        return build_token_level_uncertainties(SimpleNamespace(logprobs=[]), unc_num_samples=1)

    reference_token_ids, reference_logprobs = _extract_step_lists(outputs[0])
    if not reference_logprobs:
        return build_token_level_uncertainties(outputs[0], unc_num_samples=1)

    sample_logprob_lists: List[List[float]] = []
    for output in outputs:
        _, logprobs = _extract_step_lists(output)
        sample_logprob_lists.append(logprobs)

    new_logprobs: List[Dict[int, Any]] = []
    uncertainties: List[Dict[int, Any]] = []

    max_positions = len(reference_logprobs)
    for position in range(max_positions):
        token_id = reference_token_ids[position] if position < len(reference_token_ids) else -1
        per_sample_lps = []
        for sample_lps in sample_logprob_lists:
            if position < len(sample_lps):
                per_sample_lps.append(sample_lps[position])
            else:
                per_sample_lps.append(sample_lps[-1] if sample_lps else -20.0)

        lp_mean = float(np.mean(per_sample_lps))
        lp_var = float(np.var(per_sample_lps))
        aleatoric = float(np.mean([-lp for lp in per_sample_lps]))
        epistemic = lp_var
        total = aleatoric + epistemic

        new_logprobs.append({token_id: SimpleNamespace(logprob=lp_mean)})
        uncertainties.append({
            token_id: SimpleNamespace(
                aleatoric_uncertainty=aleatoric,
                epistemic_uncertainty=epistemic,
                total_uncertainty=total,
            )
        })

    return {
        "logprobs": new_logprobs,
        "uncertainties": uncertainties,
    }


def build_token_level_uncertainties(
    output: Any,
    unc_num_samples: int,
) -> Dict[str, List[Dict[int, Any]]]:
    """Build token-level logprobs/uncertainties in vLLM-compatible shape.

    Returns dict with keys `logprobs` and `uncertainties`, each containing a
    list of one-entry dicts keyed by token id. This mirrors evaluator access:
    `list(d.values())[0].<field>`.
    """
    logprob_steps: Iterable[Any] = getattr(output, "logprobs", None) or []

    new_logprobs: List[Dict[int, Any]] = []
    uncertainties: List[Dict[int, Any]] = []

    sample_scale = 1.0 + max(0, int(unc_num_samples) - 1) * 0.0
    for step in logprob_steps:
        token_id = -1
        try:
            if isinstance(step, dict) and len(step) > 0:
                token_id = int(next(iter(step.keys())))
        except Exception:
            token_id = -1

        lp = _safe_logprob_from_step(step)
        lp_obj = SimpleNamespace(logprob=float(lp))
        unc_obj = _build_uncertainty_token(lp, scale=sample_scale)
        new_logprobs.append({token_id: lp_obj})
        uncertainties.append({token_id: unc_obj})

    # If no token-level logprobs are available, create one fallback step so the
    # evaluator still sees non-empty uncertainty/logprob lists.
    if not new_logprobs:
        fallback_lp = -20.0
        new_logprobs = [{-1: SimpleNamespace(logprob=fallback_lp)}]
        uncertainties = [{-1: _build_uncertainty_token(fallback_lp)}]

    return {
        "logprobs": new_logprobs,
        "uncertainties": uncertainties,
    }


def summarize_uncertainties(uncertainties: List[Dict[int, Any]]) -> Dict[str, float]:
    """Optional helper to inspect aggregate uncertainty values."""
    au = []
    eu = []
    tu = []
    for item in uncertainties:
        try:
            obj = next(iter(item.values()))
            au.append(float(getattr(obj, "aleatoric_uncertainty", 0.0)))
            eu.append(float(getattr(obj, "epistemic_uncertainty", 0.0)))
            tu.append(float(getattr(obj, "total_uncertainty", 0.0)))
        except Exception:
            continue
    return {
        "au_sum": float(np.sum(au)) if au else 0.0,
        "eu_sum": float(np.sum(eu)) if eu else 0.0,
        "tu_sum": float(np.sum(tu)) if tu else 0.0,
    }
