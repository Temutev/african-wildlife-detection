# African Wildlife Detection App

A Streamlit application for detecting African wildlife (buffalo, elephant, rhino, and zebra) in images using a YOLOv11 model.

## Features

- Upload your own images or use webcam capture
- Batch process multiple images at once
- Upload and analyze video files with tracking
- Download processed images with detections
- Adjust confidence threshold for detections
- Track animal movement across video frames
- View comprehensive model analytics
- Get detailed performance metrics
- Responsive design for various screen sizes

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/ttemu/african-wildlife-detection.git
cd african-wildlife-detection
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the model
- Place your trained YOLO model file (best.pt) in the project directory
- Or update the model path in app.py to point to your model location

### 5. Run the application
```bash
streamlit run app.py
```

## Requirements

Create a file named `requirements.txt` with the following content:

```
streamlit
numpy
opencv-python
ultralytics
pillow
```

## Model Training Details

The model was trained using YOLOv11 on the African Wildlife Dataset with the following specifications:
- YOLOv11n architecture
- 100 epochs
- Image size: 640x640
- Dataset: African Wildlife (buffalo, elephant, rhino, zebra)

## Directory Structure

```
african-wildlife-detection/
├── app.py                # Main Streamlit application
├── best.pt               # Trained YOLOv8 model
├── requirements.txt      # Python dependencies
├── README.md             # This readme file
└── data.yaml             # Dataset configuration file (for reference)
```

## Further Improvements

- Add model analytics and performance metrics
- Implement video processing capabilities
- Add download option for processed images
- Deploy as a web service

## License

MIT

## Author

Tevin Temu