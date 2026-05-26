# FFN-noise-mpmi


## What is included

- `run/greedy_ffn_unc_single_batch_refine.py`: batch generation with FFN noise injection and uncertainty logging.
- `ffn_uncertainty/`: FFN patching, noise sampling, and uncertainty helpers.
- `eval/eval_detect_multi_seed.py`: multi-seed evaluation for the generated pickle files.
- `run/utils/`: shared grading and answer parsing utilities used by the evaluator.

## Main idea

The script repeats each prompt `unc_num_samples` times, injects random noise into FFN activations, and then aggregates the resulting logits into token-level uncertainty fields. The output keeps the same pickle structure expected by the existing evaluator, and each token can also carry an `mpmi` value.

## Dependencies

Create a Python environment and install the packages listed in `requirement.txt`.

The code paths in this folder use:

- `torch`
- `numpy`
- `transformers`
- `datasets`
- `tqdm`
- `click`
- `pandas`
- `scikit-learn`
- `sympy`
- `regex`
- `word2number`
- `latex2sympy2`
- `safetensors`

## Quick start

### 1. Compute uncertainty

```bash
python run/greedy_ffn_unc_single_batch_refine.py \
  --dataset-path datasets/math500.jsonl \
  --dataset-start 0 \
  --dataset-end 200 \
  --model-path /path/to/your/model \
  --output-dir ./results/debug_ffn_qwen3b_seed96/greedy_unc \
  --seed 96 \
  --batch-size 4 \
  --ffn-noise-sigma 0.05 \
  --ffn-noise-prob 0.1 \
  --ffn-noise-mode gaussian \
  --ffn-noise-layers all \
  --unc-num-samples 4
```

Useful options:

- `--ffn-noise-sigma`: Gaussian noise std.
- `--ffn-noise-prob`: neuron selection probability.
- `--ffn-noise-mode`: currently the code is written for `gaussian`.
- `--ffn-noise-layers`: layer scope, such as `all` or `0,1,2`.
- `--unc-num-samples`: number of repeated particle forwards.

The script writes:

- `batch_results_{start}_{end}.pkl`
- `ffn_method_config.json`

It also mirrors the pickle file into the evaluator-compatible `results/.../greedy_unc/` path.

### 2. Evaluate the result

```bash
python eval/eval_detect_multi_seed.py \
  --dataset math500 \
  --model debug_ffn_qwen3b_seed96 \
  --results_subdir greedy_unc \
  --seeds 96
```

## Output layout

The generation script writes result files in the same style as the original TokUR pipeline so that the existing evaluation script can read them directly.

Typical files:

- `batch_results_0_200.pkl`
- `ffn_method_config.json`

The config file records:

- `noise_sigma`
- `noise_prob`
- `noise_mode`
- `noise_layers`
- `unc_num_samples`
- `seed`
- `model_type`
- patched module count and names

## Notes from the experiment log

- FFN noise is injected only in FFN modules, not in attention projections.
- For Llama-style MLPs, the code patches `gate_proj + up_proj + down_proj` blocks.
- For Qwen-style MLPs, the code patches `gate_up_proj + down_proj` blocks.
- The uncertainty metric used by the copied evaluator is `mpmi`, so the generated pickle objects keep both `logprobs` and `uncertainties` entries.

## Reference

The implementation details were copied from the TokUR workspace and aligned with the notes in the daily run log for the FFN-only noise experiment.
