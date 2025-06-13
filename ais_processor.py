import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import cv2

class AISProcessor:
    def __init__(self, camera_params: Dict[str, float]):
        """Initialize AIS processor with camera parameters.
        
        Args:
            camera_params: Dictionary containing camera parameters
                (lon, lat, orientation, height, fov, fx, fy, u0, v0)
        """
        self.camera_params = camera_params
        self._setup_projection_matrix()
        
    def _setup_projection_matrix(self):
        """Set up the projection matrix for lat/lon to pixel conversion."""
        # Convert camera parameters to projection matrix
        # This is a simplified version - you may need to adjust based on your specific needs
        self.K = np.array([
            [self.camera_params['fx'], 0, self.camera_params['u0']],
            [0, self.camera_params['fy'], self.camera_params['v0']],
            [0, 0, 1]
        ])
        
    def load_ais_data(self, csv_path: str) -> pd.DataFrame:
        """Load and preprocess AIS data from CSV.
        
        Args:
            csv_path: Path to AIS CSV file
            
        Returns:
            DataFrame with preprocessed AIS data
        """
        df = pd.read_csv(csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
        
    def interpolate_ais_data(self, df: pd.DataFrame, target_fps: float = 10.0) -> pd.DataFrame:
        """Interpolate AIS data to fixed rate.
        
        Args:
            df: Input AIS DataFrame
            target_fps: Target frame rate for interpolation
            
        Returns:
            Interpolated DataFrame
        """
        # Create time index at target rate
        time_range = pd.date_range(
            start=df['timestamp'].min(),
            end=df['timestamp'].max(),
            freq=f'{1/target_fps}S'
        )
        
        # Interpolate each vessel's trajectory
        interpolated_data = []
        for mmsi in df['mmsi'].unique():
            vessel_data = df[df['mmsi'] == mmsi].copy()
            vessel_data.set_index('timestamp', inplace=True)
            
            # Interpolate position and motion data
            interpolated = vessel_data.reindex(time_range).interpolate(method='linear')
            interpolated['mmsi'] = mmsi
            interpolated_data.append(interpolated)
            
        return pd.concat(interpolated_data)
        
    def project_to_image(self, lat: float, lon: float) -> Tuple[float, float]:
        """Project lat/lon coordinates to image pixel coordinates.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Tuple of (x, y) pixel coordinates
        """
        # Convert lat/lon to local coordinates relative to camera
        # This is a simplified version - you'll need to implement proper
        # coordinate transformation based on your specific requirements
        dx = (lon - self.camera_params['lon']) * 111320 * np.cos(np.radians(lat))
        dy = (lat - self.camera_params['lat']) * 111320
        
        # Apply camera rotation
        angle = np.radians(self.camera_params['orientation'])
        x = dx * np.cos(angle) - dy * np.sin(angle)
        y = dx * np.sin(angle) + dy * np.cos(angle)
        
        # Project to image plane
        z = self.camera_params['height']
        point_3d = np.array([x, y, z])
        point_2d = self.K @ point_3d
        point_2d = point_2d[:2] / point_2d[2]
        
        return tuple(point_2d)
        
    def get_ais_detections(self, frame_idx: int, df: pd.DataFrame) -> List[Dict]:
        """Get AIS pseudo-detections for a specific frame.
        
        Args:
            frame_idx: Frame index
            df: Interpolated AIS DataFrame
            
        Returns:
            List of detection dictionaries with bbox and mmsi
        """
        frame_time = df.index[frame_idx]
        frame_data = df[df.index == frame_time]
        
        detections = []
        for _, row in frame_data.iterrows():
            x, y = self.project_to_image(row['latitude'], row['longitude'])
            
            # Create a pseudo-bbox around the projected point
            # Size can be adjusted based on vessel size or other factors
            bbox_size = 50  # pixels
            bbox = [
                x - bbox_size/2,
                y - bbox_size/2,
                bbox_size,
                bbox_size
            ]
            
            detections.append({
                'bbox': bbox,
                'mmsi': row['mmsi'],
                'confidence': 1.0  # AIS positions are considered ground truth
            })
            
        return detections 