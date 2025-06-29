import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple
import torch
from distance_estimator import DistanceEstimator

class VideoProcessor:
    def __init__(self, model_path: str = "yolov8x.pt", camera_params: Dict = None, conf_threshold=0.3):
        """Initialize video processor with YOLOv8 model.
        
        Args:
            model_path: Path to YOLOv8 model weights
            camera_params: Dictionary of camera parameters for distance estimation
            conf_threshold: Confidence threshold for detection
        """
        self.model = YOLO(model_path)
        self.tracks = {}  # track_id -> track_info
        self.next_track_id = 0
        
        # Initialize distance estimator if camera params are provided
        self.distance_estimator = DistanceEstimator(camera_params) if camera_params else None
        
        self.camera_params = camera_params
        self.conf_threshold = conf_threshold
        
    def process_frame(self, frame: np.ndarray, frame_idx: int, ais_data: Dict[str, Dict] = None) -> Tuple[List[Dict], Dict[int, Dict]]:
        """Process a single frame for vessel detection and tracking.
        
        Args:
            frame: Input frame
            frame_idx: Frame index
            ais_data: Dictionary mapping MMSI to AIS information (lat, lon)
            
        Returns:
            Tuple of (detections, tracks)
        """
        # Run YOLO detection
        results = self.model(frame)
        
        # Convert detections to list of dictionaries
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                if conf > self.conf_threshold:
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'frame_idx': frame_idx
                    })
        
        # Update tracks with new detections
        tracks = self._update_tracks(detections, frame_idx)
        
        # Calculate relative distances if distance estimator is available
        if self.distance_estimator is not None:
            tracks = self.distance_estimator.update_reference_ship(tracks, ais_data)
        
        # At the end, ensure tracks is a dict
        if self.tracks is None:
            self.tracks = {}
        return detections, self.tracks
    
    def _update_tracks(self, detections: List[Dict], frame_idx: int):
        """Update tracks with new detections using IoU matching.
        
        Args:
            detections: List of detection dictionaries
            frame_idx: Current frame index
        """
        # Convert detections to numpy array for easier processing
        if not detections:
            return
        
        det_boxes = np.array([d['bbox'] for d in detections])
        
        # Calculate IoU between existing tracks and new detections
        if self.tracks:
            track_boxes = np.array([t['bbox'] for t in self.tracks.values()])
            iou_matrix = np.zeros((len(self.tracks), len(detections)))
            
            for i, track_box in enumerate(track_boxes):
                for j, det_box in enumerate(det_boxes):
                    iou_matrix[i, j] = self._calculate_iou(track_box, det_box)
            
            # Match tracks to detections using IoU
            matched_tracks = set()
            matched_detections = set()
            
            # First pass: match high IoU pairs
            for i in range(len(self.tracks)):
                for j in range(len(detections)):
                    if iou_matrix[i, j] > 0.5:  # IoU threshold
                        track_id = list(self.tracks.keys())[i]
                        self.tracks[track_id]['bbox'] = det_boxes[j]
                        self.tracks[track_id]['last_seen'] = frame_idx
                        matched_tracks.add(i)
                        matched_detections.add(j)
            
            # Second pass: create new tracks for unmatched detections
            for j in range(len(detections)):
                if j not in matched_detections:
                    track_info = {
                        'bbox': det_boxes[j],
                        'last_seen': frame_idx
                    }
                    self.tracks[self.next_track_id] = track_info
                    self.next_track_id += 1
            
            # Remove old tracks
            current_tracks = list(self.tracks.keys())
            for track_id in current_tracks:
                if frame_idx - self.tracks[track_id]['last_seen'] > 30:  # Remove tracks not seen for 30 frames
                    del self.tracks[track_id]
        else:
            # Initialize tracks with all detections
            for j, det_box in enumerate(det_boxes):
                track_info = {
                    'bbox': det_box,
                    'last_seen': frame_idx
                }
                self.tracks[self.next_track_id] = track_info
                self.next_track_id += 1
        
        # Calculate relative distances if distance estimator is available
        if self.distance_estimator is not None:
            self.tracks = self.distance_estimator.update_reference_ship(self.tracks)
    
    def _calculate_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Calculate IoU between two bounding boxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
        y2 = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = bbox1[2] * bbox1[3]
        area2 = bbox2[2] * bbox2[3]
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
        
    def handle_occlusion(self, tracks: Dict[int, Dict], ais_detections: List[Dict]) -> Dict[int, Dict]:
        """Handle occlusions between tracks and AIS detections.
        
        Args:
            tracks: Dictionary of track information
            ais_detections: List of AIS detections
            
        Returns:
            Updated tracks dictionary
        """
        # For each track, check if it's occluded by any AIS detection
        for track_id, track_info in tracks.items():
            track_bbox = track_info['bbox']
            
            # Check occlusion with each AIS detection
            for ais_det in ais_detections:
                ais_bbox = ais_det['bbox']
                
                # Calculate IoU
                iou = self._calculate_iou(track_bbox, ais_bbox)
                
                # If high IoU, mark track as potentially occluded
                if iou > 0.5:
                    track_info['disappeared'] = True
                    break
            else:
                track_info['disappeared'] = False
        
        return tracks 