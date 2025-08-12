import os
import json
from flask import Flask, request, jsonify
from gradio_client import Client, handle_file
from supabase import create_client
from datetime import datetime
import tempfile

app = Flask(__name__)

# Load your credentials (use environment variables in Render)
HF_TOKEN = os.environ.get("HF_TOKEN")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

# Initialize clients
client = Client("Lightricks/ltx-video-distilled", hf_token=HF_TOKEN)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

@app.route('/generate', methods=['POST'])
def generate_video():
    try:
        # Get parameters from Flutter app
        sender_uid = request.form.get('sender_uid')
        receiver_uids = request.form.get('receiver_uids')
        prompt = request.form.get('prompt')
        
        # Get uploaded image file
        image_file = request.files.get('image')
        if not image_file:
            return jsonify({"error": "No image file provided"}), 400
        
        # Save uploaded image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_image:
            image_file.save(temp_image.name)
            temp_image_path = temp_image.name
        
        print(f"Processing request - Sender: {sender_uid}, Prompt: {prompt}")
        
        # Generate video using Hugging Face API
        result = client.predict(
            prompt=prompt,
            negative_prompt="worst quality, inconsistent motion, blurry",
            input_image_filepath=handle_file(temp_image_path),
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
        
        # Get the local video path from API response
        local_video_path = result[0]["video"]
        print(f"Video generated at: {local_video_path}")
        
        # Upload directly to Supabase videos bucket
        bucket_name = "videos"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        video_name = f"{sender_uid}_{timestamp}.mp4"
        
        # Read the generated video file and upload to Supabase
        with open(local_video_path, "rb") as video_file:
            upload_response = supabase.storage.from_(bucket_name).upload(
                video_name, 
                video_file,
                file_options={"content-type": "video/mp4"}
            )
        
        # Get public URL from Supabase
        video_public_url = f"{SUPABASE_URL}/storage/v1/object/public/{bucket_name}/{video_name}"
        
        # Clean up temporary files
        os.unlink(temp_image_path)
        if os.path.exists(local_video_path):
            os.unlink(local_video_path)
        
        # Return the public URL to Flutter app
        response_data = {
            "success": True,
            "video_url": video_public_url,
            "video_name": video_name,
            "sender_uid": sender_uid,
            "receiver_uids": receiver_uids.split(",") if receiver_uids else [],
            "prompt": prompt,
            "generated_at": timestamp
        }
        
        print(f"Video uploaded successfully: {video_public_url}")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
