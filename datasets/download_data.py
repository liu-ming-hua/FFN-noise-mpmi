from datasets import load_dataset
from pathlib import Path

REPOS = [
    "Tyrion279/math500",
    "Tyrion279/gsm8k",
    "Tyrion279/deepscaler",
    "Tyrion279/leg-counting",
    "squad",
    "allenai/nq",
]

output_dir = Path("./")
output_dir.mkdir(exist_ok=True)

for repo in REPOS:
    name = repo.split("/")[-1]
    ds = load_dataset(repo, split="train")
    ds.to_json(output_dir / f"{name}.jsonl", orient="records", lines=True)
    print(f"Saved {name}: {len(ds)} examples")