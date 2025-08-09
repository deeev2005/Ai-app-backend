import os
import uuid
import shutil
import requests
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from gradio_client import Client, handle_file
from dotenv import load_dotenv
from supabase import create_client

# Load env variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for Flutter app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your app domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Hugging Face client
client = Client("Lightricks/ltx-video-distilled", hf_token=HF_TOKEN)

# Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

@app.post("/generate/")
async def generate_video(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    friend_ids: str = Form(...)  # Comma-separated list of friend IDs
):
    try:
        # Save uploaded image temporarily
        image_id = str(uuid.uuid4())
        temp_path = f"/tmp/{image_id}.jpg"

        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Call Hugging Face model
        result = client.predict(
            prompt=prompt,
            negative_prompt="worst quality, inconsistent motion, blurry",
            input_image_filepath=handle_file(temp_path),
            input_video_filepath=None,
            height_ui=512,
            width_ui=704,
            mode="image-to-video",
            duration_ui=2,
            ui_frames_to_use=9,
            seed_ui=42,
            randomize_seed=True,
            ui_guidance_scale=1,
            improve_texture_flag=True,
            api_name="/image_to_video"
        )

        # Download the generated video from HF
        video_url = result[0]["video"]
        local_video_path = f"/tmp/{image_id}.mp4"
        video_data = requests.get(video_url).content
        with open(local_video_path, "wb") as f:
            f.write(video_data)

        # Upload to Supabase
        supabase_path = f"videos/{image_id}.mp4"
        supabase.storage.from_("videos").upload(supabase_path, local_video_path)

        # Get public URL from Supabase
        public_url = supabase.storage.from_("videos").get_public_url(supabase_path)

        # Save to Firebase (via REST API or Firestore Admin SDK)
        firebase_url = "https://firestore.googleapis.com/v1/projects/YOUR_PROJECT_ID/databases/(default)/documents/messages"
        payload = {
            "fields": {
                "video_url": {"stringValue": public_url},
                "friend_ids": {"stringValue": friend_ids},
                "prompt": {"stringValue": prompt}
            }
        }
        requests.post(firebase_url, json=payload)

        return JSONResponse({
            "supabase_url": public_url,
            "seed": result[1]
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
