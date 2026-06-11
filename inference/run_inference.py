import argparse
import cv2
from ultralytics import YOLO

def run_live_inference(model_path, source, conf_threshold, iou_threshold, img_size, device, hide_labels, hide_conf):
    """
    Runs live object detection on a camera feed using a trained YOLO model.

    Args:
        model_path (str): Path to the trained YOLOv12 model file (.pt).
        source (str or int): Camera source (e.g., '0' for webcam, RTSP/HTTP URL).
        conf_threshold (float): Confidence threshold for filtering detections.
        iou_threshold (float): IoU threshold for Non-Maximum Suppression.
        img_size (int): Image size for inference.
        device (str): Device to run inference on ('cpu', '0', '1', etc.).
        hide_labels (bool): If True, hides class labels on the output.
        hide_conf (bool): If True, hides confidence scores on the output.
    """
    # --- 1. Load the Model ---
    print(f"Loading model from: {model_path}")
    # Note: Although named YOLOv12, we use the ultralytics YOLO class to load it.
    try:
        model = YOLO(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load the model. Please check the path.")
        print(f"Details: {e}")
        return

    # --- 2. Start Prediction on the Stream ---
    print(f"Starting inference on source: {source}")
    try:
        results = model.predict(
            source=source,
            stream=True,          # Use stream mode for live feeds
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=img_size,
            device=device,
            verbose=False         # Suppress excessive logging
        )

        window_name = "YOLOv12 Live Inference"
        print("Inference started. Press 'q' to quit.")

        # --- 3. Process and Display Frames ---
        for r in results:
            # Use the built-in .plot() method to get an annotated frame (NumPy array)
            annotated_frame = r.plot(labels=not hide_labels, conf=not hide_conf)

            # Display the frame
            cv2.imshow(window_name, annotated_frame)

            # Check for 'q' key press to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nStream interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred during inference: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # --- 4. Cleanup ---
        print("Stopping stream and cleaning up...")
        cv2.destroyAllWindows()
        print("Windows closed. Exiting.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run live YOLOv12 object detection on a camera stream.")

    parser.add_argument(
        '-m', '--model',
        type=str,
        required=True,
        help="Path to your trained YOLOv12 .pt model file."
    )
    parser.add_argument(
        '-s', '--source',
        type=str,
        required=True,
        help="Camera source. Use '0' for the default webcam, or provide a URL (e.g., 'rtsp://user:pass@ip:port/stream')."
    )
    parser.add_argument(
        '-c', '--conf',
        type=float,
        default=0.3,
        help="Confidence threshold for detections (default: 0.3)."
    )
    parser.add_argument(
        '--iou',
        type=float,
        default=0.5,
        help="IoU threshold for Non-Maximum Suppression (default: 0.5)."
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help="Image size for inference (default: 640)."
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help="Device to run on, e.g., 'cpu' or '0' for GPU 0 (default: auto-detect)."
    )
    parser.add_argument(
        '--hide-labels',
        action='store_true',
        help="Hide class labels in the output."
    )
    parser.add_argument(
        '--hide-conf',
        action='store_true',
        help="Hide confidence scores in the output."
    )

    args = parser.parse_args()

    # Convert source to integer if it's a digit (for webcams)
    source_arg = int(args.source) if args.source.isdigit() else args.source

    run_live_inference(
        model_path=args.model,
        source=source_arg,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        img_size=args.imgsz,
        device=args.device,
        hide_labels=args.hide_labels,
        hide_conf=args.hide_conf
    )
