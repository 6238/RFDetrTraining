# submit_train_job.py
from google.cloud import aiplatform
from dotenv import load_dotenv

import os
from datetime import datetime

load_dotenv()

PROJECT = "visiontrainer"
LOCATION = "us-west1"

BUCKET = os.getenv("GCS_BUCKET_NAME")  # still used for staging + dataset path
DATASET_PATH = os.getenv("DATASET_PATH").split("/")[-1]

# You asked for final artifacts under: gs://visiondatabucket/models
FINAL_BUCKET = "visiondatabucket"
FINAL_PREFIX = "models"

run_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
base_output_dir = f"gs://{FINAL_BUCKET}/{FINAL_PREFIX}/rfdetr/{run_id}"

aiplatform.init(
    project=PROJECT,
    location=LOCATION,
    staging_bucket=f"gs://{BUCKET}/staging",
    experiment=input("experiment name: "),
    experiment_description=input("experiment description: "),
)

my_custom_job = aiplatform.CustomJob(
    display_name=f"rfdetr_train_{run_id}",
    worker_pool_specs=[{
        "machine_spec": {
            "machine_type": "g2-standard-8",
            "accelerator_type": "NVIDIA_L4",
            "accelerator_count": 1,
        },
        "replica_count": 1,
        "python_package_spec": {
            "executor_image_uri": "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest",
            "package_uris": [f"gs://{BUCKET}/packages/trainer-0.1.tar.gz"],
            "python_module": "trainer.train",
            "args": [
                "--dataset_uri", f"/gcs/{BUCKET}/datasets/{DATASET_PATH}.zip",
                "--learning_rate", "4e-5",
                "--epochs", "15",
                "--local_output_dir", "/job/output",
                # This tells your training code exactly where to copy artifacts:
                "--gcs_output_dir", base_output_dir,
            ],
        },
    }],
    base_output_dir=base_output_dir,  # also sets AIP_MODEL_DIR, etc. if you want them
)

my_custom_job.run(sync=True)
print("Job submitted. Artifacts target:", base_output_dir)
