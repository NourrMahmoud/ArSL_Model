import os
import random
import shutil
from pathlib import Path


# Source and destination directories - fixed path formatting for Windows
source_dir = Path('E:/ArSL_Dataset/Videos')  # Removed leading slash, using forward slashes
dest_dir = Path('E:/ArSL_Dataset/mini_dataset')  # Removed leading slash, using forward slashes

# Number of videos to select per signer per sign
NUM_VIDEOS = 10

def create_mini_dataset():
    """Create a mini dataset with 10 videos per signer per sign."""
    
    dest_dir.mkdir(exist_ok=True)
    
    # Track statistics

    
    # Iterate through sign folders
    for sign_folder in source_dir.iterdir():
        if not sign_folder.is_dir():
            continue
            
        sign_name = sign_folder.name

        logging.info(f"Processing sign: {sign_name}")
        
        # Create corresponding sign folder in mini dataset
        mini_sign_folder = dest_dir / sign_name
        mini_sign_folder.mkdir(exist_ok=True)
        
        # Iterate through signer folders
        for signer_folder in sign_folder.iterdir():
            if not signer_folder.is_dir():
                continue
                
            signer_name = signer_folder.name
            
            # Skip if the signer is ahmed 
            if signer_name.lower() == "ahmed":
                logging.info(f"Skipping signer: {signer_name}")

                continue
                

            
            # Create corresponding signer folder in mini dataset
            mini_signer_folder = mini_sign_folder / signer_name
            mini_signer_folder.mkdir(exist_ok=True)
            
            # Get all video files
            video_files = [f for f in signer_folder.iterdir() if f.is_file() and f.suffix.lower() in ['.mp4', '.avi', '.mov']]
            
            # Select videos to copy (either all if less than NUM_VIDEOS or random selection)
            if len(video_files) <= NUM_VIDEOS:
                selected_videos = video_files
            else:
                selected_videos = random.sample(video_files, NUM_VIDEOS)
            
            # Copy selected videos
            for video in selected_videos:
                dest_path = mini_signer_folder / video.name
                shutil.copy2(video, dest_path)

    


if __name__ == "__main__":
    create_mini_dataset()
