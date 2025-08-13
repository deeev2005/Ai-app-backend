async def _save_chat_messages_to_firebase(sender_uid: str, receiver_list: list, video_url: str, prompt: str):
    """Save chat messages with video URL to Firebase for each receiver"""
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore
        from datetime import datetime
        import pytz
        
        # Initialize Firebase Admin if not already done
        if not firebase_admin._apps:
            try:
                # Use the specified service account file path
                cred = credentials.Certificate("/etc/secrets/services")
                firebase_admin.initialize_app(cred)
            except Exception as e:
                logger.error(f"Failed to initialize Firebase with service account: {e}")
                raise Exception("Firebase initialization failed")
        
        db = firestore.client()
        
        # Current timestamp with timezone
        ist = pytz.timezone('Asia/Kolkata')
        timestamp = datetime.now(ist)
        
        logger.info(f"Saving video messages to Firebase for {len(receiver_list)} receivers")
        
        for receiver_id in receiver_list:
            if not receiver_id:  # Skip empty receiver IDs
                continue
                
            try:
                logger.info(f"Processing message for receiver: {receiver_id}")
                
                # Create message document with all required fields for video message
                message_data = {
                    "senderId": sender_uid,
                    "receiverId": receiver_id,
                    "text": prompt,  # The prompt as message text
                    "videoUrl": video_url,  # Supabase video URL - THIS IS KEY FOR VIDEO DISPLAY
                    "messageType": "video",  # Message type - IMPORTANT for app to know it's a video
                    "timestamp": timestamp,
                    "isRead": False,
                    "createdAt": timestamp,
                    "updatedAt": timestamp,
                    # Additional fields for better video handling
                    "hasVideo": True,  # Flag to easily identify video messages
                    "mediaType": "video",  # Explicit media type
                    "videoStatus": "uploaded"  # Status of video (uploaded, processing, failed)
                }
                
                # Add message to messages collection
                doc_ref = db.collection("messages").add(message_data)
                message_id = doc_ref[1].id
                logger.info(f"Video message saved to Firebase with ID: {message_id} for receiver {receiver_id}")
                
                # Create or update chat document
                # Use consistent chat ID format (smaller UID first)
                chat_participants = sorted([sender_uid, receiver_id])
                chat_id = f"{chat_participants[0]}_{chat_participants[1]}"
                
                # Updated chat data with video-specific fields
                chat_data = {
                    "participants": [sender_uid, receiver_id],
                    "participantIds": chat_participants,  # For easier querying
                    "lastMessage": prompt,  # Show prompt as last message preview
                    "lastMessageType": "video",  # IMPORTANT: Tells app last message was video
                    "lastMessageTimestamp": timestamp,
                    "lastSenderId": sender_uid,
                    "lastVideoUrl": video_url,  # Store last video URL for quick access
                    "lastMediaType": "video",  # Explicit media type for last message
                    "hasUnreadVideo": True,  # Flag for unread video content
                    "updatedAt": timestamp,
                    "unreadCount": {
                        receiver_id: firestore.Increment(1)  # Increment unread count for receiver
                    }
                }
                
                # Create chat if it doesn't exist, or update if it does
                chat_ref = db.collection("chats").document(chat_id)
                
                # Check if chat exists
                chat_doc = chat_ref.get()
                if chat_doc.exists:
                    # Update existing chat with video-specific fields
                    update_data = {
                        "lastMessage": prompt,
                        "lastMessageType": "video",  # Key for app to show video icon/preview
                        "lastMessageTimestamp": timestamp,
                        "lastSenderId": sender_uid,
                        "lastVideoUrl": video_url,  # URL for video preview/thumbnail
                        "lastMediaType": "video",
                        "hasUnreadVideo": True,  # Important for showing video notification
                        "updatedAt": timestamp,
                        f"unreadCount.{receiver_id}": firestore.Increment(1)
                    }
                    chat_ref.update(update_data)
                    logger.info(f"Updated existing chat with video: {chat_id}")
                else:
                    # Create new chat with video data
                    chat_data["createdAt"] = timestamp
                    chat_data["unreadCount"] = {
                        sender_uid: 0,
                        receiver_id: 1
                    }
                    chat_ref.set(chat_data)
                    logger.info(f"Created new chat with video: {chat_id}")
                
            except Exception as e:
                logger.error(f"Failed to save video message for receiver {receiver_id}: {e}")
                continue  # Continue with other receivers even if one fails
        
        logger.info("Successfully saved all video messages with URLs to Firebase")
        
    except Exception as e:
        logger.error(f"Failed to save chat messages to Firebase: {e}", exc_info=True)
        # Don't raise exception here - video generation was successful
        # Just log the error and continue
