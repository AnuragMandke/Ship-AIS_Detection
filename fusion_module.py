import numpy as np
from scipy.optimize import linear_sum_assignment
from fastdtw import fastdtw
from typing import List, Dict, Tuple
import cv2

class FusionModule:
    def __init__(self, history_length: int = 30):
        """Initialize fusion module.
        
        Args:
            history_length: Number of frames to keep in trajectory history
        """
        self.history_length = history_length
        self.track_history = {}  # track_id -> list of center points
        self.ais_history = {}    # mmsi -> list of center points
        
    def update_histories(self, tracks: Dict[int, Dict], ais_detections: List[Dict]):
        """Update trajectory histories for tracks and AIS detections.
        
        Args:
            tracks: Dictionary of track information
            ais_detections: Current AIS pseudo-detections
        """
        # Update track history
        for track_id, track_info in tracks.items():
            center = self._get_center(track_info['bbox'])
            if track_id not in self.track_history:
                self.track_history[track_id] = []
            self.track_history[track_id].append(center)
            # Keep only recent history
            if len(self.track_history[track_id]) > self.history_length:
                self.track_history[track_id] = self.track_history[track_id][-self.history_length:]
        # Update AIS history
        for det in ais_detections:
            mmsi = det['mmsi']
            center = self._get_center(det['bbox'])
            if mmsi not in self.ais_history:
                self.ais_history[mmsi] = []
            self.ais_history[mmsi].append(center)
            # Keep only recent history
            if len(self.ais_history[mmsi]) > self.history_length:
                self.ais_history[mmsi] = self.ais_history[mmsi][-self.history_length:]

    def match_trajectories(self, tracks: Dict[int, Dict], ais_detections: List[Dict]) -> Dict[int, str]:
        """Match video tracks with AIS trajectories.
        
        Args:
            tracks: Dictionary of track information
            ais_detections: Current AIS pseudo-detections
        
        Returns:
            Dictionary mapping track_id to mmsi
        """
        if not tracks or not ais_detections:
            return {}
        track_ids = list(tracks.keys())
        track_infos = list(tracks.values())
        # Build cost matrix using Fast-DTW
        cost_matrix = np.zeros((len(track_infos), len(ais_detections)))
        for i, track_info in enumerate(track_infos):
            track_id = track_ids[i]
            if track_id not in self.track_history:
                cost_matrix[i, :] = 1e6
                continue
            for j, det in enumerate(ais_detections):
                mmsi = det['mmsi']
                if mmsi not in self.ais_history:
                    cost_matrix[i, j] = 1e6
                    continue
                # Only compute DTW if both histories have at least 3 points
                if len(self.track_history[track_id]) < 3 or len(self.ais_history[mmsi]) < 3:
                    cost_matrix[i, j] = 1e6
                    continue
                try:
                    # Calculate direction-aware DTW distance
                    distance, _ = fastdtw(
                        self.track_history[track_id],
                        self.ais_history[mmsi],
                        dist=self._direction_aware_distance
                    )
                    cost_matrix[i, j] = distance
                except Exception as e:
                    cost_matrix[i, j] = 1e6
        # Solve assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        # Create mapping
        track_to_mmsi = {}
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < 1000:  # Threshold can be adjusted
                track_to_mmsi[track_ids[i]] = ais_detections[j]['mmsi']
        return track_to_mmsi
        
    def _get_center(self, bbox: List[float]) -> Tuple[float, float]:
        """Get center point of bounding box.
        
        Args:
            bbox: Bounding box [x, y, w, h]
            
        Returns:
            Center point (x, y)
        """
        return (bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2)
        
    def _direction_aware_distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate direction-aware distance between two points.
        
        Args:
            p1: First point (x, y)
            p2: Second point (x, y)
            
        Returns:
            Direction-aware distance
        """
        # Euclidean distance
        dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        
        # Add direction penalty if available
        if len(p1) > 2 and len(p2) > 2:
            angle_diff = abs(p1[2] - p2[2])
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            dist += angle_diff * 0.1  # Weight can be adjusted
            
        return dist 