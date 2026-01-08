# trainer/train.py
import argparse
import json
import os
import subprocess
from pathlib import Path

# Avoid RF-DETR accidentally going into DDP mode in single-worker Vertex runs
for k in ("RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
    os.environ.pop(k, None)
os.environ.setdefault("LOCAL_RANK", "0")

from rfdetr import RFDETRNano


def extract_last_row(obj):
    if isinstance(obj, list) and obj and isinstance(obj[-1], dict):
        return obj[-1]
    if isinstance(obj, dict):
        for k in ("history", "results", "metrics"):
            v = obj.get(k)
            if isinstance(v, list) and v and isinstance(v[-1], dict):
                return v[-1]
        return obj
    raise ValueError(f"Unrecognized results.json format: {type(obj)}")


def find_map_value(last_row: dict) -> float:
    for key in ("val_mAP", "val_map", "mAP", "map", "test_mAP", "test_map"):
        if key in last_row and last_row[key] is not None:
            return float(last_row[key])
    raise KeyError(f"Couldn't find mAP in final metrics. Available keys: {sorted(last_row.keys())}")


def gsutil_cp_recursive(src: str, dst: str):
    # Ensure trailing slash semantics are consistent
    if not dst.endswith("/"):
        dst = dst + "/"
    subprocess.run(["gsutil", "-m", "cp", "-r", src, dst], check=True)


def train_rfdetr():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--dataset_uri", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--local_output_dir", type=str, default="/job/output")
    parser.add_argument("--gcs_output_dir", type=str, default=None)  # gs://visiondatabucket/models/...
    args = parser.parse_args()

    local_data_dir = "/job/data"
    os.makedirs(local_data_dir, exist_ok=True)
    os.makedirs(args.local_output_dir, exist_ok=True)

    print("Extracting dataset...")
    subprocess.run(["unzip", "-q", args.dataset_uri, "-d", local_data_dir], check=True)

    history = []

    def on_epoch_end(data):
        history.append(data)

    model = RFDETRNano()
    model.callbacks["on_fit_epoch_end"].append(on_epoch_end)

    _ = model.train(
        dataset_dir=local_data_dir,
        output_dir=args.local_output_dir,
        lr=args.learning_rate,
        batch_size=8,
        grad_accum_steps=2,
        epochs=args.epochs,
        num_workers=1,
        # Optional if you installed metrics extras:
        # tensorboard=True,
    )

    # Read metrics from results.json written by RF-DETR
    results_path = os.path.join(args.local_output_dir, "results.json")
    final_map = None
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)
        last = extract_last_row(results)
        final_map = find_map_value(last)

    # Decide where to upload artifacts
    gcs_out = args.gcs_output_dir or os.environ.get("AIP_MODEL_DIR") or f"gs://visiondatabucket/models/rfdetr/manual"
    # Put run artifacts under a subdir to avoid collisions
    # (if user passed .../rfdetr/<run_id> already, this just adds /artifacts)
    gcs_artifacts_dir = gcs_out.rstrip("/") + "/artifacts"

    # Upload everything RF-DETR produced (checkpoints, plots, results.json, etc.)
    print(f"Uploading artifacts: {args.local_output_dir} -> {gcs_artifacts_dir}")
    gsutil_cp_recursive(args.local_output_dir + "/*", gcs_artifacts_dir)

    # Also upload a convenient "final model" filename if it exists
    best_total = Path(args.local_output_dir) / "checkpoint_best_total.pth"
    if best_total.exists():
        print(f"Uploading final checkpoint: {best_total} -> {gcs_out.rstrip('/')}/checkpoint_best_total.pth")
        subprocess.run(
            ["gsutil", "cp", str(best_total), gcs_out.rstrip("/") + "/checkpoint_best_total.pth"],
            check=True,
        )

    if final_map is not None:
        print("Final mAP:", final_map)
    else:
        print("No results.json found; skipping final mAP print.")


if __name__ == "__main__":
    train_rfdetr()
