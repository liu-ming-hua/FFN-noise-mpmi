#!/usr/bin/env python
import json
import logging
import os
import pickle
import random
import sys
import shutil
from pathlib import Path
from typing import Dict, List
from types import SimpleNamespace

import click
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from run.utils.config import Config
from ffn_uncertainty import FFNNoiseInjector
from ffn_uncertainty.config import FFNUncertaintyConfig


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_dataset_name(dataset_path: str) -> str:
    dataset_stem = Path(dataset_path).stem
    if dataset_stem.endswith(".jsonl"):
        dataset_stem = dataset_stem[:-6]
    return dataset_stem


def _get_run_name(output_dir: str, model_path: str) -> str:
    output_path = Path(output_dir).resolve()
    if output_path.parent.name:
        return output_path.parent.name
    return Path(model_path).name


def _get_eval_compat_dir(output_dir: str, model_path: str, dataset_path: str, seed: int, results_subdir: str = "greedy_unc") -> Path:
    run_name = _get_run_name(output_dir, model_path)
    dataset_name = _get_dataset_name(dataset_path)
    return Path("results") / f"{run_name}_results_vllm_pg" / dataset_name / f"seed{seed}" / results_subdir


def _save_results(all_results: List[Dict], output_dir: str, model_path: str, dataset_path: str, seed: int, start_id: int, end_id: int) -> None:
    save_name = f"batch_results_{start_id}_{end_id}.pkl"
    local_path = Path(output_dir) / save_name
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with open(local_path, "wb") as f:
        pickle.dump(all_results, f)

    compat_dir = _get_eval_compat_dir(output_dir, model_path, dataset_path, seed)
    compat_dir.mkdir(parents=True, exist_ok=True)
    compat_path = compat_dir / save_name
    if compat_path.resolve() != local_path.resolve():
        shutil.copy2(local_path, compat_path)


def _extract_question_id(sample: Dict) -> str:
    if "unique_id" in sample:
        raw_id = sample["unique_id"]
    elif "metadata" in sample and "source_index" in sample["metadata"]:
        raw_id = sample["metadata"]["source_index"]
    else:
        raise KeyError("Sample missing 'unique_id' or 'source_index' field required for identification.")
    return str(raw_id).replace("/", "_").replace(".json", "")


def _is_reading_comprehension_dataset(dataset_name: str) -> bool:
    dataset_lower = dataset_name.lower()
    return any(key in dataset_lower for key in ("squad", "nq", "natural_questions", "reading_comprehension", "rc"))


def _get_model_system_prompt(model_path: str, config: Config, dataset_name: str) -> str:
    """Get system prompt based on model type and dataset type."""
    model_lower = model_path.lower()
    if _is_reading_comprehension_dataset(dataset_name):
        return config.rc_prompt
    if "qwen" in model_lower:
        return "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    elif "llama" in model_lower:
        return config.system_prompt
    else:
        return config.system_prompt


def _format_user_message(sample: Dict, model_path: str, dataset_name: str) -> str:
    """Format user message based on model requirements and dataset type."""
    model_lower = model_path.lower()
    question = sample.get("problem", sample.get("question"))
    if question is None:
        raise KeyError("Sample missing 'problem' or 'question' field required for prompting.")

    if _is_reading_comprehension_dataset(dataset_name):
        context = sample.get("context") or sample.get("passage") or sample.get("document_text") or ""
        if context:
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nPlease answer the question based on the context."
        else:
            prompt = f"Question: {question}\n\nPlease answer the question."
    else:
        prompt = question

    if "qwen" in model_lower:
        return f"{prompt} Let's think step by step and output the final answer within \\boxed{{}}."
    return prompt


def prepare_batch_prompts(samples: List[Dict], config: Config, tokenizer, dataset_name: str = "") -> List[str]:
    """Prepare batch prompts with model-specific formatting."""
    prompts: List[str] = []
    system_prompt = _get_model_system_prompt(config.model_path, config, dataset_name)
    system = [{"role": "system", "content": system_prompt}]
    has_chat_template = bool(getattr(tokenizer, "chat_template", None))

    for sample in samples:
        user_content = _format_user_message(sample, config.model_path, dataset_name)
        if has_chat_template:
            prompt = tokenizer.apply_chat_template(
                system + [{"role": "user", "content": user_content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = f"{system_prompt}\n\n{user_content}\n\nAnswer:"
        prompts.append(prompt)
    return prompts


def _prepare_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.unk_token
    tokenizer.padding_side = "left"
    return tokenizer


def _load_model(model_path: str):
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    model.config.use_cache = True
    return model, device


def _make_logprob_entry(token_id: int, logprob: float, token_text: str) -> Dict[int, SimpleNamespace]:
    return {token_id: SimpleNamespace(logprob=float(logprob), rank=1, decoded_token=token_text)}


def _make_uncertainty_entry(token_id: int, tu: float, au: float, eu: float, token_text: str, mpmi: float = None) -> Dict[int, SimpleNamespace]:
    ns = SimpleNamespace(
        total_uncertainty=float(tu),
        aleatoric_uncertainty=float(au),
        epistemic_uncertainty=float(eu),
        bayes_implicit_reward=0.0,
        decoded_token=token_text,
    )
    # Attach token-level MeanPMI-based epistemic uncertainty if provided
    if mpmi is not None:
        try:
            ns.mpmi = float(mpmi)
        except Exception:
            ns.mpmi = None
    else:
        ns.mpmi = None
    return {token_id: ns}


def _tokenize_prompts(tokenizer, prompts: List[str], device: torch.device):
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
        add_special_tokens=False,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    return input_ids, attention_mask


def _score_and_sample_step(step_logits: torch.Tensor,
                           batch_size: int,
                           num_particles: int):
    vocab_size = step_logits.shape[-1]
    particle_logits = step_logits.reshape(batch_size, num_particles, vocab_size).float()
    particle_log_probs = torch.log_softmax(particle_logits, dim=-1)
    particle_probs = particle_log_probs.exp()

    mean_probs = particle_probs.mean(dim=1)
    mean_log_probs = torch.log(mean_probs.clamp_min(1e-12))

    # NOTE: original entropy-based uncertainty calculations are commented out
    # to only compute MeanPMI as requested. Keep placeholders to preserve
    # return signature and downstream code.
    # total_uncertainty = -(mean_probs * mean_log_probs).sum(dim=-1)
    # aleatoric_uncertainty = -(particle_probs * particle_log_probs).sum(dim=-1).mean(dim=1)
    # epistemic_uncertainty = total_uncertainty - aleatoric_uncertainty
    total_uncertainty = torch.zeros(batch_size, device=particle_probs.device)
    aleatoric_uncertainty = torch.zeros(batch_size, device=particle_probs.device)
    epistemic_uncertainty = torch.zeros(batch_size, device=particle_probs.device)

    next_token = mean_probs.argmax(dim=-1)
    next_token_logprob = mean_log_probs.gather(-1, next_token.unsqueeze(-1)).squeeze(-1)

    # Compute Mean PMI (and its negative as token-level epistemic uncertainty)
    # For each sample, gather per-particle probabilities for the chosen token
    idx = next_token.unsqueeze(-1).unsqueeze(-1).expand(batch_size, num_particles, 1)
    per_particle_p = particle_probs.gather(-1, idx).squeeze(-1)  # (batch, num_particles)
    per_particle_logp = torch.log(per_particle_p.clamp_min(1e-12))
    mean_log_pm = per_particle_logp.mean(dim=1)
    mean_p = per_particle_p.mean(dim=1)
    log_mean_p = torch.log(mean_p.clamp_min(1e-12))
    # MeanPMI = mean_log_pm - log_mean_p
    # Token-level epistemic uncertainty (U_E) = -MeanPMI = log_mean_p - mean_log_pm
    ue_mpmi = (log_mean_p - mean_log_pm)

    return next_token, next_token_logprob, total_uncertainty, aleatoric_uncertainty, epistemic_uncertainty, ue_mpmi

def batch_inference(
    samples: List[Dict],
    config: Config,
    model,
    tokenizer,
    device: torch.device,
    ffn_config: FFNUncertaintyConfig,
    prompt_type: str = "",
    dataset_name: str = "",
) -> List[Dict]:
    prompts = prepare_batch_prompts(samples, config, tokenizer, dataset_name)
    input_ids, attention_mask = _tokenize_prompts(tokenizer, prompts, device)
    batch_size = len(samples)
    num_particles = max(1, int(ffn_config.unc_num_samples))
    repeated_input_ids = input_ids.repeat_interleave(num_particles, dim=0)
    repeated_attention_mask = attention_mask.repeat_interleave(num_particles, dim=0)

    with torch.inference_mode():
        outputs = model(
            input_ids=repeated_input_ids,
            attention_mask=repeated_attention_mask,
            use_cache=True,
            return_dict=True,
        )

    past_key_values = outputs.past_key_values
    step_logits = outputs.logits[:, -1, :]
    batch_results: List[Dict] = []
    eos_token_id = tokenizer.eos_token_id
    
    # Model-specific stop tokens
    model_lower = config.model_path.lower()
    if "qwen2" in model_lower:
        # Qwen2 specific stop tokens
        additional_stop_ids = {151645, 151643}
    elif "llama" in model_lower:
        # Llama models typically use eos_token only
        additional_stop_ids = set()
    else:
        additional_stop_ids = set()
    
    stop_token_ids = set(additional_stop_ids)
    if eos_token_id is not None:
        stop_token_ids.add(int(eos_token_id))

    generated_token_ids_per_sample: List[List[int]] = [[] for _ in samples]
    logprobs_per_sample: List[List[Dict[int, SimpleNamespace]]] = [[] for _ in samples]
    uncertainties_per_sample: List[List[Dict[int, SimpleNamespace]]] = [[] for _ in samples]
    cumulative_logprob_per_sample = [0.0 for _ in samples]
    stop_reason_per_sample = [None for _ in samples]
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    current_attention_mask = repeated_attention_mask

    max_new_tokens = config.max_tokens
    for _ in range(max_new_tokens):
        next_token, next_token_logprob, tus, aus, eus, mpmi_vals = _score_and_sample_step(
            step_logits, batch_size, num_particles)

        next_token = next_token.to(device)
        if eos_token_id is not None:
            next_token = torch.where(
                finished,
                torch.full_like(next_token, int(eos_token_id)),
                next_token,
            )

        for sample_index, sample in enumerate(samples):
            if finished[sample_index]:
                continue

            token_id = int(next_token[sample_index].item())
            token_text = tokenizer.decode([token_id], skip_special_tokens=False)
            logprob_value = float(next_token_logprob[sample_index].item())
            tu_value = float(tus[sample_index].item())
            au_value = float(aus[sample_index].item())
            eu_value = float(eus[sample_index].item())
            mpmi_value = None
            try:
                mpmi_value = float(mpmi_vals[sample_index].item())
            except Exception:
                mpmi_value = None

            if token_id in stop_token_ids:
                stop_reason_per_sample[sample_index] = token_id
                finished[sample_index] = True
                continue

            generated_token_ids_per_sample[sample_index].append(token_id)
            cumulative_logprob_per_sample[sample_index] += logprob_value
            logprobs_per_sample[sample_index].append(_make_logprob_entry(token_id, logprob_value, token_text))
            uncertainties_per_sample[sample_index].append(
                _make_uncertainty_entry(token_id, tu_value, au_value, eu_value, token_text, mpmi=mpmi_value)
            )

        if bool(finished.all().item()):
            break

        repeated_next_token = next_token.repeat_interleave(num_particles).unsqueeze(-1)
        current_attention_mask = torch.cat(
            [current_attention_mask, torch.ones((current_attention_mask.shape[0], 1), device=device, dtype=current_attention_mask.dtype)],
            dim=1,
        )

        with torch.inference_mode():
            outputs = model(
                input_ids=repeated_next_token,
                attention_mask=current_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )

        past_key_values = outputs.past_key_values
        step_logits = outputs.logits[:, -1, :]

    # For each sample, pack the per-token outputs so the evaluator can use token-level data.
    for sample_index, sample in enumerate(samples):
        question_id = _extract_question_id(sample)
        prompt = prompts[sample_index]
        output = SimpleNamespace(
            text=tokenizer.decode(generated_token_ids_per_sample[sample_index], skip_special_tokens=True),
            token_ids=generated_token_ids_per_sample[sample_index],
            logprobs=logprobs_per_sample[sample_index],
            uncertainties=uncertainties_per_sample[sample_index],
            cumulative_logprob=float(cumulative_logprob_per_sample[sample_index]),
            finish_reason="length" if not finished[sample_index].item() else "stop",
            stop_reason=stop_reason_per_sample[sample_index],
        )

        result = SimpleNamespace(prompt=prompt, outputs=[output])

        batch_results.append(
            {
                "unique_id": question_id,
                "problem": sample.get("problem", sample.get("question")),
                "result": result,
                "answer": sample.get("answer"),
                "formatted_prompt": prompt,
                "old_math_orz_id": sample.get("old_math_orz_id", None),
                "level": sample.get("qwen2.5-3b-instruct-pass-at-10", None),
            }
        )
    return batch_results


@click.command()
@click.option("--dataset-path", default=None, type=str, help="Path to the dataset.", show_default=True)
@click.option("--dataset-start", default=0, type=int, help="Start index of the dataset to process.", show_default=True)
@click.option("--dataset-end", default=38, type=int, help="End index of the dataset to process.", show_default=True)
@click.option("--output-dir", default=None, type=click.Path(file_okay=False, writable=True), help="Output directory to save results.")
@click.option("--model-path", default="meta-llama/Llama-3.2-1B-Instruct", type=str, help="Path to the language model.", show_default=True)
@click.option("--seed", default=96, type=int, help="Random seed for reproducibility.", show_default=True)
@click.option("--batch-size", default=8, type=int, help="Batch size for inference.", show_default=True)
@click.option("--save-every-n-batches", default=10, type=int, help="Save results every N batches.", show_default=True)
@click.option("--ffn-noise-sigma", default=0.05, type=float, help="FFN Gaussian noise std.", show_default=True)
@click.option("--ffn-noise-prob", default=0.1, type=float, help="FFN neuron noise probability.", show_default=True)
@click.option("--ffn-noise-mode", default="gaussian", type=str, help="FFN noise mode.", show_default=True)
@click.option("--ffn-noise-layers", default="all", type=str, help="FFN layer selection (e.g., 'all' or '0,1,2').", show_default=True)
@click.option("--unc-num-samples", default=4, type=int, help="Uncertainty sampling count.", show_default=True)
def main(
    dataset_path: str,
    dataset_start: int,
    dataset_end: int,
    output_dir: str,
    model_path: str,
    seed: int,
    batch_size: int,
    save_every_n_batches: int,
    ffn_noise_sigma: float,
    ffn_noise_prob: float,
    ffn_noise_mode: str,
    ffn_noise_layers: str,
    unc_num_samples: int,
) -> None:
    if output_dir is None:
        raise ValueError("--output-dir is required")
    if dataset_path is None:
        raise ValueError("--dataset-path is required")

    set_global_seed(seed)

    config = Config()
    config.dataset_start = dataset_start
    config.dataset_end = dataset_end
    config.output_dir = output_dir
    config.model_path = model_path
    os.makedirs(config.output_dir, exist_ok=True)

    ffn_config = FFNUncertaintyConfig(
        noise_sigma=ffn_noise_sigma,
        noise_prob=ffn_noise_prob,
        noise_mode=ffn_noise_mode,
        noise_layers=ffn_noise_layers,
        unc_num_samples=unc_num_samples,
        seed=seed,
    )

    logger.info("SEED=%s", seed)
    logger.info("BATCH_SIZE=%s", batch_size)
    logger.info("FFN_CONFIG=%s", ffn_config)

    os.environ["TOKUR_FFN_NOISE_ENABLED"] = "1"
    os.environ["TOKUR_FFN_NOISE_SIGMA"] = str(ffn_config.noise_sigma)
    os.environ["TOKUR_FFN_NOISE_PROB"] = str(ffn_config.noise_prob)
    os.environ["TOKUR_FFN_NOISE_MODE"] = ffn_config.noise_mode
    os.environ["TOKUR_FFN_NOISE_LAYERS"] = ffn_config.noise_layers
    tokenizer = _prepare_tokenizer(config.model_path)
    model, device = _load_model(config.model_path)

    injector = FFNNoiseInjector(ffn_config)
    injector.apply(model)
    metadata = injector.metadata()
    logger.info("FFN injector initialized")
    logger.info("  Model Type: %s", metadata.get("model_type", "unknown"))
    logger.info("  Patched FFN Modules: %d", metadata.get("num_patched_modules", 0))
    logger.info("  FFN Config: %s", ffn_config)

    with open(os.path.join(config.output_dir, "ffn_method_config.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.select(range(config.dataset_start, config.dataset_end))

    total_samples = len(dataset)
    num_batches = (total_samples + batch_size - 1) // batch_size
    logger.info("Processing %s samples in %s batches", total_samples, num_batches)

    all_results: List[Dict] = []
    batch_save_counter = 0

    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_samples)
        current_batch = dataset.select(range(start_idx, end_idx))
        batch_samples = [sample for sample in current_batch]

        try:
            batch_results = batch_inference(
                batch_samples,
                config=config,
                model=model,
                tokenizer=tokenizer,
                device=device,
                ffn_config=ffn_config,
                prompt_type=config.model_path.lower().split("/")[-1],
                dataset_name=_get_dataset_name(dataset_path),
            )
            all_results.extend(batch_results)
        except Exception as e:
            logger.error("Error processing batch %s: %s", batch_idx, e)
            continue

        if (batch_idx + 1) % save_every_n_batches == 0 or batch_idx == num_batches - 1:
            start_id = config.dataset_start + batch_save_counter * save_every_n_batches * batch_size
            end_id = config.dataset_start + len(all_results)
            _save_results(
                all_results,
                output_dir=config.output_dir,
                model_path=config.model_path,
                dataset_path=dataset_path,
                seed=seed,
                start_id=start_id,
                end_id=end_id,
            )
            logger.info("Saved %s samples to %s", len(all_results), os.path.join(config.output_dir, f"batch_results_{start_id}_{end_id}.pkl"))

            compat_dir = _get_eval_compat_dir(config.output_dir, config.model_path, dataset_path, seed)
            logger.info("Mirrored results to %s", compat_dir)

            all_results = []
            batch_save_counter += 1

    logger.info("FFN uncertainty stage-1 inference completed")


if __name__ == "__main__":
    main()
