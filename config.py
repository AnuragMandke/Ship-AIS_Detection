import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.absolute()
DATASET_DIR = BASE_DIR / "FVessel" / "01_Video+AIS"  # Updated dataset directory
MODELS_DIR = BASE_DIR / "models"

# Create models directory if it doesn't exist
MODELS_DIR.mkdir(exist_ok=True)

class DatasetConfig:
    def __init__(self):
        self.dataset_dir = DATASET_DIR
        self.video_extensions = ['.mp4']  # Supported video formats
        self.ais_extensions = ['.csv']    # Supported AIS data formats
        self.camera_params_extensions = ['.txt']  # Supported camera parameter formats
        
        # Model paths
        self.model_dir = MODELS_DIR
        self.yolo_model_path = self.model_dir / "yolov8x.pt"

    def get_video_folders(self):
        """Get list of all video folders."""
        return [d for d in self.dataset_dir.iterdir() if d.is_dir() and d.name.startswith('Video-')]

    def set_video_path(self, video_folder: str):
        """Set the path for the video file in a specific folder.
        
        Args:
            video_folder: Name of the video folder (e.g., 'Video-01')
        """
        folder_path = self.dataset_dir / video_folder
        if not folder_path.exists():
            raise ValueError(f"Video folder {video_folder} not found")
            
        # Find the video file in the folder
        for ext in self.video_extensions:
            video_files = list(folder_path.glob(f"*{ext}"))
            if video_files:
                return str(video_files[0])
        raise ValueError(f"No video file found in {video_folder}")

    def set_ais_path(self, video_folder: str):
        """Set the path for the AIS data directory in a specific folder.
        
        Args:
            video_folder: Name of the video folder (e.g., 'Video-01')
        """
        folder_path = self.dataset_dir / video_folder / "ais"
        if not folder_path.exists():
            raise ValueError(f"AIS directory not found in {video_folder}")
        return str(folder_path)

    def set_camera_params_path(self, video_folder: str):
        """Set the path for the camera parameters file in a specific folder.
        
        Args:
            video_folder: Name of the video folder (e.g., 'Video-01')
        """
        folder_path = self.dataset_dir / video_folder
        if not folder_path.exists():
            raise ValueError(f"Video folder {video_folder} not found")
            
        # Find the camera parameters file
        for ext in self.camera_params_extensions:
            param_files = list(folder_path.glob(f"*{ext}"))
            if param_files:
                return str(param_files[0])
        raise ValueError(f"No camera parameters file found in {video_folder}")

    def get_all_paths(self):
        """Get all configured paths."""
        return {
            "dataset_dir": str(self.dataset_dir),
            "model_dir": str(self.model_dir),
            "yolo_model_path": str(self.yolo_model_path)
        }

    def list_available_videos(self):
        """List all available video folders and their contents."""
        videos = {}
        for folder in self.get_video_folders():
            folder_name = folder.name
            videos[folder_name] = {
                "video": None,
                "ais_files": [],
                "camera_params": None
            }
            
            # Get video file
            for ext in self.video_extensions:
                video_files = list(folder.glob(f"*{ext}"))
                if video_files:
                    videos[folder_name]["video"] = video_files[0].name
                    break
                    
            # Get AIS files
            ais_dir = folder / "ais"
            if ais_dir.exists():
                videos[folder_name]["ais_files"] = [f.name for f in ais_dir.glob("*.csv")]
                
            # Get camera parameters
            for ext in self.camera_params_extensions:
                param_files = list(folder.glob(f"*{ext}"))
                if param_files:
                    videos[folder_name]["camera_params"] = param_files[0].name
                    break
                    
        return videos

# Create a global instance
dataset_config = DatasetConfig()

# Example usage:
if __name__ == "__main__":
    print("Dataset directory:", str(DATASET_DIR))
    print("\nAvailable videos:")
    videos = dataset_config.list_available_videos()
    for folder, contents in videos.items():
        print(f"\n{folder}:")
        print(f"  Video: {contents['video']}")
        print(f"  Camera parameters: {contents['camera_params']}")
        print(f"  AIS files: {len(contents['ais_files'])} files") 