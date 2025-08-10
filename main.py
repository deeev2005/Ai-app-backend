import os
import uuid
import shutil
import tempfile
import json
import requests
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from gradio_client import Client, handle_file
from dotenv import load_dotenv

# Supabase
from supabase import create_client as create_supabase_client

# Firebase Admin
import firebase_admin
from firebase_admin import credentials, firestore

# Load local .env when running locally
load_dotenv()

# --- Environment Variables ---
HF_TOKEN = os.getenv("HF_TOKEN")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GAC_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")  # Path to service account JSON

# Validation
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN not set")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set")
if not GAC_PATH:
    raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS must be set")

# Remove any accidental newline characters
GAC_PATH = GAC_PATH.strip()

# Initialize Supabase
supabase_client = create_supabase_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize Firebase Admin
if not firebase_admin._apps:
    cred = credentials.Certificate(GAC_PATH)
    firebase_admin.initialize_app(cred)
db = firestore.client()

# Initialize FastAPI
app = FastAPI(title="AI Video Generator - Supabase + Firestore")

# Allow CORS (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Hugging Face Client
client = Client("Lightricks/ltx-video-distilled", hf_token=HF_TOKEN)


def upload_to_supabase(local_path: str, dest_folder: str = "videos") -> str:
    """Uploads file to Supabase Storage and returns a public URL."""
    filename = os.path.basename(local_path)
    dest_path = f"{uuid.uuid4().hex}_{filename}"

    with open(local_path, "rb") as f:
        supabase_client.storage.from_(dest_folder).upload(dest_path, f)

    # Try to get public URL
    try:
        pub = supabase_client.storage.from_(dest_folder).get_public_url(dest_path)
        return (
            pub.get("publicURL")
            or pub.get("publicUrl")
            or pub.get("public_url")
            or f"{SUPABASE_URL}/storage/v1/object/public/{dest_folder}/{dest_path}"
        )
    except Exception:
        return f"{SUPABASE_URL}/storage/v1/object/public/{dest_folder}/{dest_path}"


@app.post("/generate/")
async def generate_video(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    sender_uid: str = Form(...),
    friend_uids: str = Form(...),  # comma-separated UIDs
):
    """
    Accepts an image and prompt, generates a video using HF,
    uploads it to Supabase, and sends Firestore messages.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        # 1) Save incoming image
        input_path = os.path.join(temp_dir, f"{uuid.uuid4().hex}_{file.filename}")
        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # 2) Call Hugging Face model
        try:
            result = client.predict(
                prompt=prompt,
                negative_prompt="worst quality, inconsistent motion, blurry",
                input_image_filepath=handle_file(input_path),
                input_video_filepath=None,
                height_ui=640,
                width_ui=352,
                mode="image-to-video",
                duration_ui=5,
                ui_frames_to_use=9,
                seed_ui=42,
                randomize_seed=True,
                ui_guidance_scale=5,
                improve_texture_flag=True,
                api_name="/image_to_video",
            )
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"HF model call failed: {str(e)}"})

        # 3) Extract video from model output
        video_ref = None
        if isinstance(result, list) and result:
            first = result[0]
            if isinstance(first, dict):
                video_ref = first.get("video") or first.get("output")
            elif isinstance(first, str):
                video_ref = first

        local_video_path = None
        if isinstance(video_ref, str) and video_ref.startswith("http"):
            r = requests.get(video_ref, timeout=120)
            if r.status_code == 200:
                local_video_path = os.path.join(temp_dir, f"out_{uuid.uuid4().hex}.mp4")
                with open(local_video_path, "wb") as out:
                    out.write(r.content)
            else:
                return JSONResponse(status_code=500, content={"error": "Failed to download HF output"})
        elif isinstance(video_ref, (bytes, bytearray)):
            local_video_path = os.path.join(temp_dir, f"out_{uuid.uuid4().hex}.mp4")
            with open(local_video_path, "wb") as out:
                out.write(video_ref)
        elif isinstance(video_ref, str) and os.path.exists(video_ref):
            local_video_path = video_ref
        else:
            return JSONResponse(status_code=500, content={"error": "Unexpected HF response", "result": result})

        if not local_video_path:
            return JSONResponse(status_code=500, content={"error": "No video output produced"})

        # 4) Upload video to Supabase
        try:
            public_url = upload_to_supabase(local_video_path, dest_folder="videos")
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"Supabase upload failed: {str(e)}"})

        # 5) Write Firestore messages for each friend
        friend_list = [f.strip() for f in friend_uids.split(",") if f.strip()]
        for friend_uid in friend_list:
            chat_id = f"{sender_uid}_{friend_uid}" if sender_uid < friend_uid else f"{friend_uid}_{sender_uid}"
            chat_ref = db.collection("chats").document(chat_id)

            chat_ref.set({
                "participants": [sender_uid, friend_uid],
                "lastMessage": "[AI Video]",
                "lastMessageTime": firestore.SERVER_TIMESTAMP
            }, merge=True)

            chat_ref.collection("messages").add({
                "senderId": sender_uid,
                "receiverId": friend_uid,
                "videoUrl": public_url,
                "prompt": prompt,
                "type": "video",
                "timestamp": firestore.SERVER_TIMESTAMP,
            })

        return JSONResponse(status_code=200, content={"status": "success", "public_video_url": public_url})

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


