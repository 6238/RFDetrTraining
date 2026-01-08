import os
import shutil
from google.cloud import storage
from dotenv import load_dotenv

load_dotenv()

def prepare_and_upload():
    bucket_name = os.getenv("GCS_BUCKET_NAME")
    dataset_path = os.getenv("DATASET_PATH") # e.g., './my_coco_data'
    zip_filename = dataset_path.split("/")[-1]
    
    # 1. Create Zip (retains structure: train/, val/, annotations/)
    print(f"Zipping {dataset_path}...")
    shutil.make_archive(zip_filename, 'zip', dataset_path)
    
    # 2. Upload to GCS
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"datasets/{zip_filename}.zip")
    
    print(f"Uploading to gs://{bucket_name}/datasets/{zip_filename}.zip...")
    blob.upload_from_filename(f"{zip_filename}.zip")
    print("Upload complete.")

if __name__ == "__main__":
    prepare_and_upload()