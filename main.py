import os
import uuid
import shutil
import supabase
import firebase_admin
from firebase_admin import credentials, firestore
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from gradio_client import Client, handle_file
from dotenv import load_dotenv

# Load env
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Supabase client
supabase_client = supabase.create_client(SUPABASE_URL, SUPABASE_KEY)

# Firebase admin
cred = credentials.Certificate("firebase-key.json")  # service account JSON
firebase_admin.initialize_app(cred)
db = firestore.client()

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# HF client
client = Client("Lightricks/ltx-video-distilled", hf_token=HF_TOKEN)


@app.post("/generate/")
async def generate_video(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    sender_uid: str = Form(...),
    friend_uids: str = Form(...)  # comma-separated
):
    try:
        # Save uploaded image temporarily
        image_id = str(uuid.uuid4())
        temp_path = f"/tmp/{image_id}.jpg"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Call HF model (image-to-video)
        result = client.predict(
            prompt=prompt,
            negative_prompt="worst quality, inconsistent motion, blurry",
            input_image_filepath=handle_file(temp_path),
            input_video_filepath=None,
            height_ui=1920,
            width_ui=1080,
            mode="image-to-video",
            duration_ui=5,
            ui_frames_to_use=9,
            seed_ui=42,
            randomize_seed=True,
            ui_guidance_scale=5,
            improve_texture_flag=True,
            api_name="/image_to_video"
        )

        video_url = result[0]["video"]

        # Download the output video from HF
        local_video_path = f"/tmp/{uuid.uuid4()}.mp4"
        import requests
        vid_resp = requests.get(video_url)
        with open(local_video_path, "wb") as f:
            f.write(vid_resp.content)

        # Upload to Supabase storage
        supabase_path = f"videos/{uuid.uuid4()}.mp4"
        supabase_client.storage.from_("videos").upload(supabase_path, local_video_path)
        public_url = supabase_client.storage.from_("videos").get_public_url(supabase_path)

        # Save to Firestore for each friend
        for friend_uid in friend_uids.split(","):
            chat_id = (
                f"{sender_uid}_{friend_uid}"
                if sender_uid < friend_uid
                else f"{friend_uid}_{sender_uid}"
            )

            chat_ref = db.collection("chats").document(chat_id)
            chat_ref.set({
                "participants": [sender_uid, friend_uid],
                "lastUpdated": firestore.SERVER_TIMESTAMP
            }, merge=True)

            chat_ref.collection("messages").add({
                "senderId": sender_uid,
                "videoUrl": public_url,
                "timestamp": firestore.SERVER_TIMESTAMP,
                "type": "video"
            })

        return JSONResponse({
            "status": "success",
            "public_video_url": public_url
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
