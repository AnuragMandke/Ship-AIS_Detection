import os
import pandas as pd
from pathlib import Path
import cv2
import time
from fusion_pipeline import fusion_pipeline
from config import dataset_config

def prepare_subset_data(video_folder: str, num_frames: int = 100):
    """Prepare a subset of data for testing.
    
    Args:
        video_folder: Name of the video folder (e.g., 'Video-01')
        num_frames: Number of frames to process
    """
    # Get paths
    video_path = dataset_config.set_video_path(video_folder)
    ais_dir = dataset_config.set_ais_path(video_folder)
    camera_params_path = dataset_config.set_camera_params_path(video_folder)
    
    # Create temporary directory for subset data
    temp_dir = Path("temp_data")
    temp_dir.mkdir(exist_ok=True)
    
    # Copy video file and trim it
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer
    output_video = str(temp_dir / "subset_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Read and write frames
    frame_count = 0
    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frame_count += 1
    
    # Release video resources
    cap.release()
    out.release()
    
    # Process AIS data
    ais_files = sorted(list(Path(ais_dir).glob("*.csv")))
    if ais_files:
        # Read first few AIS files
        subset_ais = pd.concat([pd.read_csv(f) for f in ais_files[:5]])
        # Rename columns for consistency
        subset_ais = subset_ais.rename(columns={"lon": "longitude", "lat": "latitude"})
        # Drop duplicates based on timestamp and mmsi
        if 'timestamp' in subset_ais.columns and 'mmsi' in subset_ais.columns:
            subset_ais = subset_ais.drop_duplicates(subset=['timestamp', 'mmsi'])
        subset_ais.to_csv(temp_dir / "subset_ais.csv", index=False)
    
    # Copy camera parameters
    import shutil
    shutil.copy(camera_params_path, temp_dir / "camera_para.txt")
    
    return {
        "video_path": output_video,
        "ais_path": str(temp_dir / "subset_ais.csv"),
        "camera_params": str(temp_dir / "camera_para.txt")
    }

def cleanup_temp_data():
    """Clean up temporary data files."""
    # Ensure all OpenCV windows are closed
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # Give time for windows to close
    
    # Wait a bit to ensure resources are released
    time.sleep(1)
    
    import shutil
    temp_dir = Path("temp_data")
    if temp_dir.exists():
        try:
            shutil.rmtree(temp_dir)
        except PermissionError:
            print("Warning: Could not delete some temporary files. They may be in use.")
            print("Please close any open video windows and try again.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run pipeline with subset of data")
    parser.add_argument("video_folder", help="Video folder name (e.g., 'Video-01')")
    parser.add_argument("--frames", type=int, default=100, help="Number of frames to process")
    parser.add_argument("--output", type=str, default="output_frames", help="Directory to save output frames")
    
    args = parser.parse_args()
    
    try:
        print(f"Preparing subset of data from {args.video_folder}...")
        paths = prepare_subset_data(args.video_folder, args.frames)
        
        print("\nRunning pipeline with subset data:")
        print(f"Video: {paths['video_path']}")
        print(f"AIS data: {paths['ais_path']}")
        print(f"Camera parameters: {paths['camera_params']}")
        print(f"Output directory: {args.output}")
        
        fusion_pipeline(paths['video_path'], paths['ais_path'], paths['camera_params'], output_dir=args.output)
        print(f"\nDetection images saved to: {args.output}")
        
    finally:
        print("\nCleaning up temporary data...")
        cleanup_temp_data() 