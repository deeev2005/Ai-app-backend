import os
import json
import time
import tempfile
import shutil
import subprocess
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from kaggle.api.kaggle_api_extended import KaggleApi

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Kaggle credentials from env or mounted secret
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY = os.getenv("KAGGLE_KEY")
KERNEL_SLUG = os.getenv("KERNEL_SLUG")  # e.g. "yourusername/your-kernel-name"

# Setup Kaggle API client
def setup_kaggle_api():
    # Write kaggle.json file for kaggle client
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    with open(os.path.join(kaggle_dir, "kaggle.json"), "w") as f:
        json.dump({"username": KAGGLE_USERNAME, "key": KAGGLE_KEY}, f)
    os.chmod(os.path.join(kaggle_dir, "kaggle.json"), 0o600)
    api = KaggleApi()
    api.authenticate()
    return api

@app.post("/generate/")
async def generate_video(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    sender_uid: str = Form(...),
    friend_uids: str = Form(...),
):
    temp_dir = tempfile.mkdtemp()
    try:
        input_path = os.path.join(temp_dir, file.filename)
        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        api = setup_kaggle_api()

        # Upload input file to Kaggle dataset or to somewhere accessible if needed
        # For simplicity, let's assume kernel accesses input from a public URL or you pre-upload input elsewhere.

        # Start kernel run with parameters passed via kernel metadata or environment variables
        # Note: Kaggle API does not support passing arbitrary params directly to kernel runs
        # So you may store inputs in a dataset, or use environment variables/secrets in kernel

        # Trigger kernel run (this call blocks until complete)
        print("Starting Kaggle kernel run...")
        api.kernels_run(kernel=KERNEL_SLUG, wait=True, quiet=False)
        print("Kernel run finished.")

        # After run, output file should be saved in a persistent place (Supabase) by kernel itself
        # Here, you might retrieve output metadata or have your kernel write result URL to a dataset or Firestore

        # For demo, assume you get output URL from Firestore or your own method
        # Return dummy URL for now
        public_video_url = "https://your.supabase.storage/videos/generated_video.mp4"

        return JSONResponse(content={"status": "success", "public_video_url": public_video_url})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
