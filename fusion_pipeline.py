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
        
        # Get relative distance if available
        relative_distance = track_info.get('relative_distance', None)
        distance_type = track_info.get('distance_type', 'pixel')
        
        # Set color based on whether this is the reference ship
        is_reference = track_info.get('is_reference', False)
        color = (0, 0, 255) if is_reference else (0, 255, 0)  # Red for reference, green for others
        
        # Draw bounding box
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
        
        # Create label with track ID, MMSI, and relative distance
        label = f"Track {track_id} (MMSI: {mmsi})"
        if relative_distance is not None:
            if is_reference:
                label += " [REF]"
            else:
                if distance_type == 'ais':
                    # Convert meters to kilometers for display
                    distance_km = relative_distance / 1000
                    label += f" | {distance_km:.1f}km from ref"
                else:
                    label += f" | {relative_distance:.0f}px from ref"
        
        # Draw label
        cv2.putText(vis_frame, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw distance lines between ships
        if not is_reference and relative_distance is not None:
            ref_track = next((t for t in tracks.values() if t.get('is_reference', False)), None)
            if ref_track:
                ref_bbox = ref_track['bbox']
                ref_center = (
                    int((ref_bbox[0] + ref_bbox[2]) / 2),
                    int((ref_bbox[1] + ref_bbox[3]) / 2)
                )
                curr_center = (
                    int((x1 + x2) / 2),
                    int((y1 + y2) / 2)
                )
                cv2.line(vis_frame, ref_center, curr_center, (255, 255, 0), 1)
    
    return vis_frame

def fusion_pipeline(video_source: str, ais_stream: str, camera_params: str, output_dir: str = "output_frames"):
    """Main fusion pipeline function.
    
    Args:
        video_source: Path to video file
        ais_stream: Path to AIS data CSV
        camera_params: Path to camera parameters file
        output_dir: Directory to save output frames and video
    """
    # Initialize components
    camera_params_dict = load_camera_params(camera_params)
    ais_processor = AISProcessor(camera_params_dict)
    video_processor = VideoProcessor(camera_params=camera_params_dict)
    fusion_module = FusionModule()
    
    # Load and preprocess AIS data
    ais_data = ais_processor.load_ais_data(ais_stream)
    ais_data = ais_processor.interpolate_ais_data(ais_data)
    
    # Open video
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_source}")
    
    # Get video properties for output
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Prepare output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize video writer
    video_output_path = os.path.join(output_dir, 'output_video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))
    if not video_writer.isOpened():
        raise ValueError(f"Could not create video writer: {video_output_path}")
    
    print(f"Processing video...")
    print(f"Output video will be saved to: {video_output_path}")
    
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
        
        # Write frame to video
        video_writer.write(vis_frame)
        
        # Print progress every 10 frames
        if frame_idx % 10 == 0:
            print(f"Processed frame {frame_idx}/{num_ais_frames}")
        
        frame_idx += 1
    
    # Release resources
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    
    print(f"\nProcessing complete!")
    print(f"Detection images saved to: {output_dir}")
    print(f"Output video saved to: {video_output_path}")

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