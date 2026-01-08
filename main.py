from google.cloud import aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt

from dotenv import load_dotenv
import os
load_dotenv()

BUCKET = os.getenv("GCS_BUCKET_NAME")
DATASET_PATH = os.getenv("DATASET_PATH").split("/")[-1]

aiplatform.init(
    project="visiontrainer",
    location="us-west1",
    staging_bucket=f"gs://{BUCKET}/staging",
    experiment=input("expierment name:"),
    experiment_description=input("expierment description:"),
)

my_custom_job = aiplatform.CustomJob(
    display_name="rfdetr_trial_template",
    worker_pool_specs=[{
        "machine_spec": {
            "machine_type": "g2-standard-8",
            "accelerator_type": "NVIDIA_L4",
            "accelerator_count": 1,
        },
        "replica_count": 1,
        "python_package_spec": {
            # Prebuilt training container:
            "executor_image_uri": "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest",

            # Your sdist/wheel in GCS:
            "package_uris": [f"gs://{BUCKET}/packages/trainer-0.1.tar.gz"],

            # Module to run (python -m trainer.train):
            "python_module": "trainer.train",

            # Base args for your module; Vertex HPT appends tuned params:
            "args": [
                "--dataset_uri", f"/gcs/{BUCKET}/datasets/{DATASET_PATH}.zip",
            ],
        },
    }],
)

parameter_spec = {
    "learning_rate": hpt.DoubleParameterSpec(min=1e-5, max=1e-3, scale="log"),
}

metric_spec = {"mean_average_precision": "maximize"}

tuning_job = aiplatform.HyperparameterTuningJob(
    display_name="rfdetr_hpt_run",
    custom_job=my_custom_job,
    metric_spec=metric_spec,
    parameter_spec=parameter_spec,
    max_trial_count=10,
    parallel_trial_count=4,
)

tuning_job.run()
