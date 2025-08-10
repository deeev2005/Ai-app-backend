import os
import uuid
import shutil
import tempfile
import json
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from gradio_client import Client, handle_file
from dotenv import load_dotenv
import requests

# Supabase client
from supabase import create_client as create_supabase_client

# Firebase Admin
import firebase_admin
from firebase_admin import credentials, firestore

# load .env (optional)
load_dotenv()

# --- Environment variables (must be set in Render) ---
HF_TOKEN = os.getenv("HF_TOKEN")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GAC_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")  # e.g. /etc/secrets/service_account.json

if HF_TOKEN is None:
    raise RuntimeError("HF_TOKEN not set")
if SUPABASE_URL is None or SUPABASE_KEY is None:
    raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set")
if GAC_PATH is None:
    raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS must be set to your service account JSON path")

# Initialize Supabase
supabase_client = create_supabase_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize Firebase Admin (using the mounted secret file)
if not firebase_admin._apps:
    cred = credentials.Certificate(GAC_PATH)
    firebase_admin.initialize_app(cred)
db = firestore.client()

# Initialize FastAPI
app = FastAPI(title="AI -> Supabase -> Firestore Bridge")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gradio/HF client
client = Client("Lightricks/ltx-video-distilled", hf_token=HF_TOKEN)


def upload_to_supabase(local_path: str, dest_folder: str = "videos") -> str:
    """
    Upload a file to Supabase storage and return a public URL.
    """
    filename = os.path.basename(local_path)
    dest_path = f"{dest_folder}/{uuid.uuid4().hex}_{filename}"

    # open file as bytes
    with open(local_path, "rb") as f:
        res = supabase_client.storage.from_(dest_folder).upload(dest_path, f)
    # res may contain info or error; attempt to get public URL
    try:
        pub = supabase_client.storage.from_(dest_folder).get_public_url(dest_path)
        # different supabase client versions return different keys
        public_url = (
            pub.get("publicURL")
            or pub.get("publicUrl")
            or pub.get("public_url")
            or None
        )
        if public_url:
            return public_url
    except Exception:
        pass

    # If the SDK didn't return a proper public url, construct a URL:
    # NOTE: this works if your bucket is public; otherwise you'll need signed URLs.
    # The following pattern is typical for Supabase (storage API):
    # https://<SUPABASE_URL>/storage/v1/object/public/<bucket>/<path>
    try:
        base = SUPABASE_URL.rstrip("/")
        constructed = f"{base}/storage/v1/object/public/{dest_folder}/{dest_path}"
        return constructed
    except Exception:
        raise RuntimeError("Unable to determine Supabase public URL for uploaded file")


@app.post("/generate/")
async def generate_video(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    sender_uid: str = Form(...),
    friend_uids: str = Form(...),  # comma-separated UIDs
):
    """
    Accepts image + prompt, generates video (via HF space), uploads to Supabase,
    and writes message docs in Firestore for each friend.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        # 1) Save incoming image locally
        input_filename = f"{uuid.uuid4().hex}_{file.filename}"
        input_path = os.path.join(temp_dir, input_filename)
        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # 2) Call Hugging Face / Gradio space
        try:
            # reduce sizes/duration to avoid long times; tune as needed
            result = client.predict(
                prompt=prompt,
                negative_prompt="worst quality, inconsistent motion, blurry",
                input_image_filepath=handle_file(input_path),
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
                api_name="/image_to_video",
            )
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"HF model call failed: {str(e)}"})

        # 3) Extract video reference from result
        video_ref = None
        if isinstance(result, list) and len(result) > 0:
            first = result[0]
            if isinstance(first, dict):
                video_ref = first.get("video") or first.get("output") or None
            elif isinstance(first, str):
                video_ref = first

        # If we have an HTTP URL, download it; otherwise if it's a local path or bytes, handle accordingly
        local_video_path = None
        if isinstance(video_ref, str) and video_ref.startswith("http"):
            # download remote file
            try:
                r = requests.get(video_ref, timeout=120)
                if r.status_code == 200:
                    local_video_path = os.path.join(temp_dir, f"out_{uuid.uuid4().hex}.mp4")
                    with open(local_video_path, "wb") as out:
                        out.write(r.content)
                else:
                    return JSONResponse(status_code=500, content={"error": f"Failed to download HF output: status {r.status_code}"})
            except Exception as e:
                return JSONResponse(status_code=500, content={"error": f"Download failed: {str(e)}"})
        elif isinstance(video_ref, (bytes, bytearray)):
            local_video_path = os.path.join(temp_dir, f"out_{uuid.uuid4().hex}.mp4")
            with open(local_video_path, "wb") as out:
                out.write(video_ref)
        elif isinstance(video_ref, str) and os.path.exists(video_ref):
            local_video_path = video_ref
        else:
            # fallback: maybe the model returned a direct file-like blob elsewhere in result
            # Return the whole result for debugging
            return JSONResponse(status_code=500, content={"error": "Unexpected HF response", "result": result})

        if local_video_path is None:
            return JSONResponse(status_code=500, content={"error": "No video output produced by model"})

        # 4) Upload to Supabase storage
        try:
            public_url = upload_to_supabase(local_video_path, dest_folder="videos")
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"Supabase upload failed: {str(e)}"})

        # 5) Write to Firestore messages for each friend
        friend_list = [f.strip() for f in friend_uids.split(",") if f.strip()]
        for friend_uid in friend_list:
            # deterministic chat ID
            chat_id = f"{sender_uid}_{friend_uid}" if sender_uid < friend_uid else f"{friend_uid}_{sender_uid}"
            chat_ref = db.collection("chats").document(chat_id)

            # ensure chat doc exists/updated
            chat_ref.set({
                "participants": [sender_uid, friend_uid],
                "lastMessage": "[AI Video]",
                "lastMessageTime": firestore.SERVER_TIMESTAMP
            }, merge=True)

            # add message
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
        # cleanup temp files
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass
