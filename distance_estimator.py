import numpy as np
from typing import Dict, Tuple, List
import cv2

class DistanceEstimator:
    def __init__(self, camera_params: Dict[str, float]):
        """Initialize distance estimator with camera parameters.
        
        Args:
            camera_params: Dictionary containing camera parameters
                - latitude, longitude: Camera position
                - height: Camera height above sea level
                - orientation: Camera orientation in degrees
                - fov: Field of view in degrees
                - fx, fy: Focal lengths
                - u0, v0: Principal point
        """
        self.camera_params = camera_params
        self.focal_length = (camera_params['fx'] + camera_params['fy']) / 2
        self.principal_point = (camera_params['u0'], camera_params['v0'])
        
        # Convert orientation to radians
        self.orientation_rad = np.radians(camera_params['orientation'])
        
        # Calculate camera matrix
        self.camera_matrix = np.array([
            [camera_params['fx'], 0, camera_params['u0']],
            [0, camera_params['fy'], camera_params['v0']],
            [0, 0, 1]
        ])
        
    def estimate_distance(self, bbox: List[float], image_size: Tuple[int, int]) -> float:
        """Estimate distance to vessel using geometric method.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            image_size: (width, height) of the image
            
        Returns:
            Estimated distance in meters
        """
        # Get bbox center and size
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        
        # Calculate angle from principal point
        dx = center_x - self.principal_point[0]
        dy = center_y - self.principal_point[1]
        angle = np.arctan2(dy, dx)
        
        # Calculate distance using similar triangles
        # Assuming average vessel width of 20 meters (can be adjusted)
        avg_vessel_width = 20.0  # meters
        distance = (avg_vessel_width * self.focal_length) / bbox_width
        
        return distance
    
    def estimate_distance_with_ais(self, bbox: List[float], 
                                 ais_position: Tuple[float, float]) -> float:
        """Estimate distance using AIS data for calibration.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            ais_position: (latitude, longitude) of vessel from AIS
            
        Returns:
            Estimated distance in meters
        """
        # Convert AIS position to image coordinates
        ais_x, ais_y = self._project_ais_to_image(ais_position)
        
        # Calculate distance between bbox center and AIS projection
        x1, y1, x2, y2 = bbox
        bbox_center = ((x1 + x2) / 2, (y1 + y2) / 2)
        
        # Use this to calibrate the geometric distance estimate
        geometric_distance = self.estimate_distance(bbox, (1920, 1080))  # Assuming 1080p
        calibration_factor = self._calculate_calibration_factor(
            bbox_center, (ais_x, ais_y), geometric_distance)
        
        return geometric_distance * calibration_factor
    
    def _project_ais_to_image(self, ais_position: Tuple[float, float]) -> Tuple[float, float]:
        """Project AIS coordinates to image coordinates.
        
        Args:
            ais_position: (latitude, longitude) of vessel
            
        Returns:
            (x, y) coordinates in image space
        """
        # Convert lat/lon to local coordinates
        local_x, local_y = self._latlon_to_local(ais_position)
        
        # Project to image plane
        point_3d = np.array([local_x, local_y, 0])
        point_2d, _ = cv2.projectPoints(
            point_3d, 
            np.zeros(3),  # rotation vector
            np.zeros(3),  # translation vector
            self.camera_matrix,
            None
        )
        
        return tuple(point_2d[0][0])
    
    def _latlon_to_local(self, position: Tuple[float, float]) -> Tuple[float, float]:
        """Convert latitude/longitude to local coordinates.
        
        Args:
            position: (latitude, longitude) of point
            
        Returns:
            (x, y) in local coordinate system
        """
        # Convert to radians
        lat1, lon1 = np.radians(self.camera_params['latitude']), np.radians(self.camera_params['longitude'])
        lat2, lon2 = np.radians(position[0]), np.radians(position[1])
        
        # Calculate local coordinates
        R = 6371000  # Earth radius in meters
        x = R * np.cos(lat1) * (lon2 - lon1)
        y = R * (lat2 - lat1)
        
        return (x, y)
    
    def _calculate_calibration_factor(self, 
                                   bbox_center: Tuple[float, float],
                                   ais_projection: Tuple[float, float],
                                   geometric_distance: float) -> float:
        """Calculate calibration factor based on AIS position.
        
        Args:
            bbox_center: Center of bounding box
            ais_projection: Projected AIS position
            geometric_distance: Initial geometric distance estimate
            
        Returns:
            Calibration factor
        """
        # Calculate distance between bbox center and AIS projection
        dx = bbox_center[0] - ais_projection[0]
        dy = bbox_center[1] - ais_projection[1]
        pixel_distance = np.sqrt(dx*dx + dy*dy)
        
        # Simple calibration based on pixel distance
        # This can be improved with more sophisticated methods
        return 1.0 / (1.0 + pixel_distance / 1000.0)  # Normalize by 1000 pixels 