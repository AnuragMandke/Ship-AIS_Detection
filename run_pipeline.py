import os
import pandas as pd
from pathlib import Path
import cv2
import time
from fusion_pipeline import fusion_pipeline
from config import dataset_config
import argparse
import glob

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

def main():
    parser = argparse.ArgumentParser(description='Run Ship+AIS Detection Pipeline')
    parser.add_argument('video_name', help='Name of the video file (without extension)')
    parser.add_argument('--frames', type=int, default=50, help='Number of frames to process')
    parser.add_argument('--output', default='output_frames', help='Output directory for frames')
    args = parser.parse_args()
    
    # Get paths from config
    video_folder = args.video_name
    video_dir = os.path.join(dataset_config.base_path, video_folder)
    video_files = glob.glob(os.path.join(video_dir, '*.mp4'))
    if not video_files:
        raise FileNotFoundError(f"No video file found in {video_dir}")
    video_path = video_files[0]
    
    ais_dir = dataset_config.set_ais_path(video_folder)
    camera_params_path = dataset_config.set_camera_params_path(video_folder)
    
    # Find the first CSV file in the AIS directory
    ais_csv_files = glob.glob(os.path.join(ais_dir, '*.csv'))
    if not ais_csv_files:
        raise FileNotFoundError(f"No AIS CSV files found in {ais_dir}")
    ais_path = ais_csv_files[0]
    
    # Run pipeline
    fusion_pipeline(
        video_source=video_path,
        ais_stream=ais_path,
        camera_params=camera_params_path,
        output_dir=args.output
    )

if __name__ == "__main__":
    main() 