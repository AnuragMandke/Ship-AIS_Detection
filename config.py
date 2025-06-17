import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.absolute()
MODELS_DIR = BASE_DIR / "models"

# Create models directory if it doesn't exist
MODELS_DIR.mkdir(exist_ok=True)

class DatasetConfig:
    def __init__(self):
        # Update base path to new location as a Path object
        self.base_path = Path(r"E:/Projects/FVessel/01_Video+AIS")
        self.video_extensions = ['.mp4']  # Supported video formats
        self.ais_extensions = ['.csv']    # Supported AIS data formats
        self.camera_params_extensions = ['.txt']  # Supported camera parameter formats
        
        # Model paths
        self.model_dir = MODELS_DIR
        self.yolo_model_path = self.model_dir / "yolov8x.pt"

    def get_video_folders(self):
        """Get list of all video folders."""
        return [d for d in self.base_path.iterdir() if d.is_dir() and d.name.startswith('Video-')]

    def set_video_path(self, video_folder: str) -> str:
        """Set path to video file."""
        return str(self.base_path / video_folder / f"{video_folder}.mp4")
    
    def set_ais_path(self, video_folder: str) -> str:
        """Set path to AIS data directory."""
        return str(self.base_path / video_folder / "AIS")
    
    def set_camera_params_path(self, video_folder: str) -> str:
        """Set path to camera parameters file."""
        return str(self.base_path / video_folder / "camera_para.txt")

    def get_all_paths(self):
        """Get all configured paths."""
        return {
            "dataset_dir": str(self.base_path),
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
            ais_dir = folder / "AIS"
            if ais_dir.exists():
                videos[folder_name]["ais_files"] = [f.name for f in ais_dir.glob("*.csv")]
                
            # Get camera parameters
            for ext in self.camera_params_extensions:
                param_files = list(folder.glob(f"*{ext}"))
                if param_files:
                    videos[folder_name]["camera_params"] = param_files[0].name
                    break
                    
        return videos

# Create global instance
dataset_config = DatasetConfig()

# Example usage:
if __name__ == "__main__":
    print("Dataset directory:", str(dataset_config.base_path))
    print("\nAvailable videos:")
    videos = dataset_config.list_available_videos()
    for folder, contents in videos.items():
        print(f"\n{folder}:")
        print(f"  Video: {contents['video']}")
        print(f"  Camera parameters: {contents['camera_params']}")
        print(f"  AIS files: {len(contents['ais_files'])} files") 