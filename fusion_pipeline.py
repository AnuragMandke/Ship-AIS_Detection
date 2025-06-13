import cv2
import numpy as np
from typing import Dict, List, Tuple
import json
from ais_processor import AISProcessor
from video_processor import VideoProcessor
from fusion_module import FusionModule
from config import dataset_config
import os

def load_camera_params(params_path: str) -> Dict[str, float]:
    """Load camera parameters from file.
    
    Args:
        params_path: Path to camera parameters file
        
    Returns:
        Dictionary of camera parameters
    """
    with open(params_path, 'r') as f:
        # Remove brackets and split by comma
        params = f.read().strip().strip('[]').split(',')
        
    return {
        'lon': float(params[0]),
        'lat': float(params[1]),
        'orientation': float(params[2]),
        'height': float(params[3]),
        'fov': float(params[4]),
        'fx': float(params[5]),
        'fy': float(params[6]),
        'u0': float(params[7]),
        'v0': float(params[8])
    }

def visualize_frame(frame: np.ndarray, tracks: Dict[int, Dict], track_to_mmsi: Dict[int, str]) -> np.ndarray:
    """Visualize detection and tracking results on frame.
    
    Args:
        frame: Input frame
        tracks: Dictionary of track information
        track_to_mmsi: Mapping of track IDs to MMSI numbers
        
    Returns:
        Frame with visualizations
    """
    vis_frame = frame.copy()
    
    # Draw tracks and their labels
    for track_id, track_info in tracks.items():
        # Get bounding box
        bbox = track_info['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        
        # Get MMSI if available
        mmsi = track_to_mmsi.get(track_id, 'Unknown')
        
        # Draw bounding box
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label with track ID and MMSI
        label = f"Track {track_id} (MMSI: {mmsi})"
        cv2.putText(vis_frame, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return vis_frame

def fusion_pipeline(video_source: str, ais_stream: str, camera_params: str, output_dir: str = "output_frames"):
    """Main fusion pipeline function.
    
    Args:
        video_source: Path to video file
        ais_stream: Path to AIS data CSV
        camera_params: Path to camera parameters file
        output_dir: Directory to save output frames
    """
    # Initialize components
    camera_params_dict = load_camera_params(camera_params)
    ais_processor = AISProcessor(camera_params_dict)
    video_processor = VideoProcessor()
    fusion_module = FusionModule()
    
    # Load and preprocess AIS data
    ais_data = ais_processor.load_ais_data(ais_stream)
    ais_data = ais_processor.interpolate_ais_data(ais_data)
    
    # Open video
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_source}")
        
    # Prepare output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine the minimum number of frames to process
    num_ais_frames = len(ais_data.index)
    frame_idx = 0
    while True:
        if frame_idx >= num_ais_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get AIS detections for current frame
        ais_detections = ais_processor.get_ais_detections(frame_idx, ais_data)
        
        # Process video frame
        detections, tracks = video_processor.process_frame(frame, frame_idx)
        
        # Handle occlusions
        tracks = video_processor.handle_occlusion(tracks, ais_detections)
        
        # Update trajectory histories
        fusion_module.update_histories(tracks, ais_detections)
        
        # Match trajectories
        track_to_mmsi = fusion_module.match_trajectories(tracks, ais_detections)
        
        # Visualize results
        vis_frame = visualize_frame(frame, tracks, track_to_mmsi)
        
        # Save the frame as an image
        out_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.jpg")
        cv2.imwrite(out_path, vis_frame)
        
        # Do NOT display results
        # cv2.imshow('Fusion Results', vis_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        
        frame_idx += 1
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AIS+Video Fusion Pipeline")
    parser.add_argument("video_folder", help="Video folder name (e.g., 'Video-01')")
    
    args = parser.parse_args()
    
    # Get paths from config
    video_path = dataset_config.set_video_path(args.video_folder)
    ais_dir = dataset_config.set_ais_path(args.video_folder)
    camera_params_path = dataset_config.set_camera_params_path(args.video_folder)
    
    print(f"Processing video folder: {args.video_folder}")
    print(f"Video file: {video_path}")
    print(f"AIS directory: {ais_dir}")
    print(f"Camera parameters: {camera_params_path}")
    
    # Run pipeline
    fusion_pipeline(video_path, ais_dir, camera_params_path) 