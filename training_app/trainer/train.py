import argparse
import hypertune # Required for reporting metrics
from rfdetr import RFDETRNano
import os 
import subprocess

def train_rfdetr():
    parser = argparse.ArgumentParser()
    # Parameters to tune
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--dataset_uri', type=str)
    args = parser.parse_args()

    local_data_dir = "/job/data"
    os.makedirs(local_data_dir, exist_ok=True)

    # 2. Download and Unzip from GCS
    # Vertex AI includes gsutil by default
    
    print("Extracting dataset...")
    subprocess.run(["unzip", "-q", args.dataset_uri, "-d", local_data_dir], check=True)

    model = RFDETRNano()
    history = model.train(
        dataset_dir="/job/data",
        lr=args.learning_rate,
        batch_size=args.batch_size
    )

    # Report final validation mAP to Vertex AI
    # RF-DETR history usually contains 'val_mAP'
    final_map = history[-1]['val_mAP']
    
    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='mean_average_precision',
        metric_value=final_map,
        global_step=len(history)
    )

if __name__ == "__main__":
    train_rfdetr()