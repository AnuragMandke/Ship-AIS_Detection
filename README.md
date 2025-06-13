# Ship+AIS Detection

A computer vision pipeline for detecting ships in video and fusing the detections with AIS (Automatic Identification System) data.

## Features

- Ship detection using YOLOv8
- AIS data processing and interpolation
- Trajectory tracking and matching
- Occlusion handling
- Visualization of detections and tracks

## Requirements

- Python 3.8+
- OpenCV
- PyTorch
- Ultralytics YOLOv8
- Pandas
- NumPy
- FastDTW
- scipy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Ship+AIS_Detection.git
cd Ship+AIS_Detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download YOLOv8 model:
```bash
# The model will be downloaded automatically on first run
# or you can download it manually and place it in the models directory
```

## Usage

Run the pipeline on a video folder:

```bash
python run_pipeline.py Video-01 --frames 50 --output output_frames
```

Arguments:
- `video_folder`: Name of the video folder (e.g., 'Video-01')
- `--frames`: Number of frames to process (default: 100)
- `--output`: Directory to save output frames (default: 'output_frames')

## Project Structure

- `run_pipeline.py`: Main script to run the pipeline
- `video_processor.py`: Video processing and object detection
- `ais_processor.py`: AIS data processing and interpolation
- `fusion_module.py`: Trajectory matching and fusion
- `fusion_pipeline.py`: Main pipeline implementation
- `config.py`: Configuration and path management

## License

MIT License
