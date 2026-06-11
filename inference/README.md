
# Live YOLO Object Detection Inference

This script provides a simple command-line interface to run live object detection on various video streams using a trained YOLO model from the Ultralytics framework. It is designed for easy deployment and testing of your custom-trained `.pt` models on webcams, IP cameras, or other video URLs.

---

## ⚙️ Requirements

Ensure you have the necessary Python libraries installed. You can install them using `pip`:

```bash
pip install ultralytics opencv-python
```

---

## 🚀 How to Use

The script is run directly from your terminal. The two mandatory arguments are the path to your trained model (`--model`) and the video source (`--source`).

### Basic Command Structure

```bash
python live_inference.py --model path/to/your/best.pt --source <camera_source>
```

A live video window will appear showing the object detection results. **Press the 'q' key** while the window is active to quit the stream.

### Command-Line Examples

#### 1. Using a Local Webcam
To use your computer's default webcam (usually identified as device `0`):```bash
python live_inference.py --model path/to/your/best.pt --source 0
```

#### 2. Using a Network IP Camera (RTSP Stream)
For a camera that provides a standard RTSP stream:
```bash
python live_inference.py --model path/to/your/best.pt --source "rtsp://username:password@192.168.1.108:554/stream1"
```

#### 3. Using a Public HTTP Video Stream
For a public video stream from a URL:
```bash
python live_inference.py --model path/to/your/best.pt --source "http://some-website.com/live/stream.m3u8"
```

#### 4. With Custom Options
To run with a higher confidence threshold (e.g., 50%) and hide the confidence scores from the display:
```bash
python live_inference.py --model path/to/your/best.pt --source 0 --conf 0.5 --hide-conf
```

---

## 🛠️ Command-Line Arguments

| Argument | Short Form | Description | Default |
| :--- | :--- | :--- | :--- |
| `--model` | `-m` | **(Required)** Path to your trained YOLO `.pt` model file. | - |
| `--source` | `-s` | **(Required)** Camera source. Use `0`, `1`... for webcams, or a full URL for network streams. | - |
| `--conf` | `-c` | The confidence threshold for filtering detections. Only boxes with a score higher than this value will be shown. | `0.3` |
| `--iou` | - | The IoU (Intersection over Union) threshold for Non-Maximum Suppression. | `0.5` |
| `--imgsz` | - | The image size (width/height in pixels) to use for inference. | `640` |
| `--device` | - | The device to run inference on. Examples: `cpu`, `0` (for GPU 0), `mps` (for Apple Silicon). | `auto` |
| `--hide-labels` | - | A flag to hide the class name labels from the output video. | `False` |
| `--hide-conf` | - | A flag to hide the confidence scores from the output video. | `False` |

---

## 💡 How It Knows the Class Names

You do not need to provide a list of class names to this script. The class names are **automatically read from the `.pt` model file itself**. This information is embedded in the model during the training process from your `dataset.yaml` file, making inference simple and self-contained.