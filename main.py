import os
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from gradio_client import Client, handle_file
from dotenv import load_dotenv

# Load HF token from .env
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

app = FastAPI()

# Enable CORS (allow Flutter app to connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can limit this to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gradio client once
client = Client("Lightricks/ltx-video-distilled", hf_token=HF_TOKEN)

@app.post("/generate/")
async def generate_video(
    file: UploadFile = File(...),
    prompt: str = Form(...)
):
    try:
        # Save uploaded image temporarily
        image_id = str(uuid.uuid4())
        temp_path = f"/tmp/{image_id}.jpg"

        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Call HF model
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

        video_url = result[0]["video"]
        seed_used = result[1]

        return JSONResponse({
            "video_url": video_url,
            "seed": seed_used
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
