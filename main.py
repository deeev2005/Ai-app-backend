import os
import uuid
import shutil
import asyncio
import logging
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from gradio_client import Client, handle_file
from dotenv import load_dotenv
from supabase import create_client, Client as SupabaseClient
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_KEY")  # Use service key for server-side operations

if not HF_TOKEN:
    logger.error("HF_TOKEN not found in environment variables")
    raise ValueError("HF_TOKEN is required")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    logger.error("SUPABASE_URL or SUPABASE_SERVICE_KEY not found in environment variables")
    raise ValueError("Supabase credentials are required")

app = FastAPI(title="AI Video Generator", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global clients
client = None
supabase: SupabaseClient = None

@app.on_event("startup")
async def startup_event():
    global client, supabase
    try:
        logger.info("Initializing Gradio client...")
        client = Client("Lightricks/ltx-video-distilled", hf_token=HF_TOKEN)
        logger.info("Gradio client initialized successfully")
        
        logger.info("Initializing Supabase client...")
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        logger.info("Supabase client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize clients: {e}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "client_ready": client is not None,
        "supabase_ready": supabase is not None
    }

@app.post("/generate/")
async def generate_video(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    sender_uid: str = Form(...),
    receiver_uids: str = Form(...)
):
    """Generate video from image and prompt"""
    temp_image_path = None
    temp_video_path = None
    
    try:
        # Improved image validation
        content_type = file.content_type or ""
        filename = file.filename or ""
        
        # Check content type OR file extension
        valid_content_types = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp']
        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        
        is_valid_content_type = any(content_type.startswith(ct) for ct in ['image/jpeg', 'image/jpg', 'image/png', 'image/webp', 'image/'])
        is_valid_extension = any(filename.lower().endswith(ext) for ext in valid_extensions)
        
        if not (is_valid_content_type or is_valid_extension):
            logger.warning(f"Invalid file - Content-Type: {content_type}, Filename: {filename}")
            raise HTTPException(status_code=400, detail="File must be an image (jpg, png, webp)")
        
        if len(prompt.strip()) == 0:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")

        logger.info(f"Starting video generation for user {sender_uid}")
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Receivers: {receiver_uids}")
        logger.info(f"File info - Content-Type: {content_type}, Filename: {filename}")

        # Create temp directory if it doesn't exist
        temp_dir = Path("/tmp")
        temp_dir.mkdir(exist_ok=True)

        # Save uploaded image temporarily
        image_id = str(uuid.uuid4())
        file_extension = Path(filename).suffix or '.jpg'
        temp_image_path = temp_dir / f"{image_id}{file_extension}"

        # Save file
        with open(temp_image_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        logger.info(f"Image saved to {temp_image_path}")

        # Validate file size (optional)
        file_size = temp_image_path.stat().st_size
        if file_size > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=400, detail="File too large (max 10MB)")

        # Check if clients are available
        if client is None:
            raise HTTPException(status_code=503, detail="AI service not available")
        
        if supabase is None:
            raise HTTPException(status_code=503, detail="Storage service not available")

        # Call HF model with timeout
        logger.info("Calling Hugging Face model...")
        
        # Run the prediction with asyncio timeout
        result = await asyncio.wait_for(
            asyncio.to_thread(_predict_video, str(temp_image_path), prompt),
            timeout=300.0  # 5 minutes timeout
        )

        if not result or len(result) < 2:
            raise HTTPException(status_code=500, detail="Invalid response from AI model")

        local_video_path = result[0].get("video") if isinstance(result[0], dict) else result[0]
        seed_used = result[1] if len(result) > 1 else "unknown"

        logger.info(f"Video generated locally: {local_video_path}")

        # Upload video to Supabase storage
        video_url = await _upload_video_to_supabase(local_video_path, sender_uid)
        
        logger.info(f"Video uploaded to Supabase: {video_url}")

        # Save chat messages to Firebase for each receiver
        receiver_list = receiver_uids.split(",")
        await _save_chat_messages_to_firebase(sender_uid, receiver_list, video_url, prompt)

        return JSONResponse({
            "success": True,
            "video_url": video_url,
            "seed": seed_used,
            "sender_uid": sender_uid,
            "receiver_uids": receiver_list
        })

    except asyncio.TimeoutError:
        logger.error("Video generation timed out after 5 minutes")
        raise HTTPException(
            status_code=408, 
            detail="Video generation timed out. Please try with a simpler prompt or smaller image."
        )
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        logger.error(f"Error generating video: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate video: {str(e)}"
        )
    
    finally:
        # Cleanup temporary files
        for temp_path in [temp_image_path, temp_video_path]:
            if temp_path and Path(temp_path).exists():
                try:
                    Path(temp_path).unlink()
                    logger.info(f"Cleaned up temp file: {temp_path}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp file {temp_path}: {e}")

async def _upload_video_to_supabase(local_video_path: str, sender_uid: str) -> str:
    """Upload video to Supabase storage and return public URL"""
    try:
        video_path = Path(local_video_path)
        if not video_path.exists():
            raise Exception(f"Video file not found: {local_video_path}")

        # Generate unique filename for Supabase storage
        video_id = str(uuid.uuid4())
        storage_path = f"videos/{sender_uid}/{video_id}.mp4"

        # Read video file
        with open(video_path, "rb") as video_file:
            video_data = video_file.read()

        logger.info(f"Uploading video to Supabase: {storage_path}")

        # Upload to Supabase storage
        try:
            result = supabase.storage.from_("videos").upload(
                path=storage_path,
                file=video_data,
                file_options={
                    "content-type": "video/mp4",
                    "cache-control": "3600"
                }
            )
            logger.info(f"Upload result: {result}")
            
        except Exception as upload_error:
            logger.error(f"Upload failed: {upload_error}")
            raise Exception(f"Supabase upload failed: {upload_error}")

        # Get public URL
        try:
            url_result = supabase.storage.from_("videos").get_public_url(storage_path)
            logger.info(f"Generated public URL: {url_result}")
            
            if not url_result:
                raise Exception("Failed to get public URL")
            
            return url_result
            
        except Exception as url_error:
            logger.error(f"Failed to get public URL: {url_error}")
            raise Exception(f"Failed to get public URL: {url_error}")

            return url_result
            
        except Exception as url_error:
            logger.error(f"Failed to get public URL: {url_error}")
            raise Exception(f"Failed to get public URL: {url_error}")

    except Exception as e:
        logger.error(f"Failed to upload video to Supabase: {e}")
        raise Exception(f"Storage upload failed: {str(e)}")

async def _save_chat_messages_to_firebase(sender_uid: str, receiver_list: list, video_url: str, prompt: str):
    """Save chat messages with video URL to Firebase for each receiver"""
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore
        from datetime import datetime
        
        # Initialize Firebase Admin (add your service account key)
        if not firebase_admin._apps:
            # You need to add your Firebase service account JSON file
            cred = credentials.Certificate("/etc/secrets/services")
            firebase_admin.initialize_app(cred)
        
        db = firestore.client()
        
        # Current timestamp
        timestamp = datetime.now()
        
        for receiver_id in receiver_list:
            receiver_id = receiver_id.strip()  # Clean whitespace
            
            # Create or get chat document ID (consistent format)
            chat_participants = sorted([sender_uid, receiver_id])
            chat_id = f"{chat_participants[0]}_{chat_participants[1]}"
            
            # Create message document
            message_data = {
                "senderId": sender_uid,
                "receiverId": receiver_id,
                "text": prompt,
                "videoUrl": video_url,  # Add video URL field
                "messageType": "video",  # Add message type
                "timestamp": timestamp,
                "isRead": False
            }
            
            # Add message to messages collection
            message_ref = db.collection("messages").add(message_data)
            logger.info(f"Message saved to Firebase for receiver {receiver_id}: {message_ref[1].id}")
            
            # Update or create chat document
            chat_ref = db.collection("chats").document(chat_id)
            chat_data = {
                "participants": [sender_uid, receiver_id],
                "lastMessage": prompt,
                "lastMessageType": "video",
                "lastMessageTimestamp": timestamp,
                "lastSenderId": sender_uid
            }
            
            chat_ref.set(chat_data, merge=True)
            logger.info(f"Chat updated for chat_id: {chat_id}")
            
    except Exception as e:
        logger.error(f"Failed to save chat messages to Firebase: {e}")
        # Don't raise exception here - video generation was successful
        # Just log the error and continue

def _predict_video(image_path: str, prompt: str):
    """Synchronous function to call the Gradio client"""
    try:
        return client.predict(
            prompt=prompt,
            negative_prompt="worst quality, inconsistent motion, blurry, artifacts",
            input_image_filepath=handle_file(image_path),
            input_video_filepath=None,
            height_ui=512,
            width_ui=704,
            mode="image-to-video",
            duration_ui=5,
            ui_frames_to_use=9,
            seed_ui=42,
            randomize_seed=True,
            ui_guidance_scale=5,
            improve_texture_flag=True,
            api_name="/image_to_video"
        )
    except Exception as e:
        logger.error(f"Gradio client prediction failed: {e}")
        raise

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception handler caught: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        timeout_keep_alive=300,  # 5 minutes keep alive
        timeout_graceful_shutdown=30
    )


