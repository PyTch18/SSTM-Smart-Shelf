import time
from ultralytics import YOLO  # Updated import
import cv2
from collections import defaultdict
import requests
import json
import os

# Define mapping from raw YOLO names to the 6 real money classes
CLASS_MAPPING = {
    "0": "10_EGP",
    "1": "5_EGP",
    "10": "50_EGP",
    "11": "10_EGP",
    "12": "10_EGP",
    "13": "20_EGP",
    "3": "20_EGP",
    "8": "5_EGP",
    "5": "200_EGP",
    "4": "200_EGP",
    "6": "50_EGP",
    "7": "20_EGP",
    "2": "20_EGP",
    "9": "100_EGP",
    # already correct ones should map to themselves
    "100_EGP": "100_EGP",
    "10_EGP": "10_EGP",
    "200_EGP": "200_EGP",
    "20_EGP": "20_EGP",
    "50_EGP": "50_EGP",
    "5_EGP": "5_EGP",
}

def money_detector_counter(image_path, model):
    print(f"[INFO] Detecting objects in frame: {image_path}")

    results = model(image_path)
    class_counts = defaultdict(int)

    for result in results:
        boxes = result.boxes
        names = result.names  # class name dictionary

        for box in boxes:
            cls_id = int(box.cls)
            class_name = names[cls_id]
            class_counts[class_name] += 1

    return class_counts


def video_frame_distribution(video_path, model_path="yolov8n.pt"):
    print(f"[INFO] Loading YOLO model from: {model_path}")
    model = YOLO(model_path)

    print(f"[INFO] Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    frame_index = 0
    total_distribution = defaultdict(int)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Finished reading all frames.")
            break

        if frame_index % 5 == 0:
            print(f"[INFO] Processing frame {frame_index}")
            temp_path = "temp_frame.jpg"
            cv2.imwrite(temp_path, frame)

            frame_distribution = money_detector_counter(temp_path, model)

            for money_type, count in frame_distribution.items():
                total_distribution[money_type] += count

            os.remove(temp_path)  # Clean up temp file

        frame_index += 1

    cap.release()
    print("[INFO] Video processing complete.")
    return dict(total_distribution)


def send_to_thingsboard(distribution_dict, token, tb_url="http://localhost:8080"):
    print("[INFO] Sending data to ThingsBoard...")

    url = f"{tb_url}/api/v1/{token}/telemetry"
    headers = {"Content-Type": "application/json"}
    payload = json.dumps(distribution_dict)

    response = requests.post(url, headers=headers, data=payload)

    if response.status_code == 200:
        print("[SUCCESS] Data sent to ThingsBoard successfully.")
    else:
        print(f"[ERROR] Failed to send data. Status code: {response.status_code}")
        print("Response:", response.text)

def display_frame(frame, frame_index):
    """Displays the given frame using OpenCV. Press 'q' to stop early."""
    window_name = f"Processing Frame {frame_index}"
    cv2.imshow(window_name, frame)
    print(f"[INFO] Displaying frame {frame_index} — press 'q' to stop early.")

    key = cv2.waitKey(500) & 0xFF  # Show for 500ms or until key press
    cv2.destroyWindow(window_name)

    return key != ord('q')  # Return False if 'q' is pressed

def list_available_cameras(max_tested=5):
    available = []
    for i in range(max_tested):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"✅ Camera found at index {i}")
            available.append(i)
            cap.release()
    return available


def main():
    # Load YOLO model
    model = YOLO("detection_model.pt")

    # Open external webcam
    cams = list_available_cameras()
    if not cams:
        print("Error: No camera found.")
        return

    #For the external webcam choose index 0
    #If this failed, please try other indexes
    print(f"Available cameras: {cams}")
    camera_index = int(input("👉 Enter the camera index you want to use: "))
    cap = cv2.VideoCapture(camera_index)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    recording = False
    out = None
    last_detection_time = 0  # <-- track last detection timestamp

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame.")
            break

        # Run YOLO detection
        results = model(frame, imgsz=720)
        detections = results[0].boxes

        # Debug
        print(f"Detections: {len(detections)}")

        # If there’s a detection, update last_detection_time
        if len(detections) > 0:
            last_detection_time = time.time()
            if not recording:
                print("🎥 Detection found! Start recording...")
                recording = True
                out = cv2.VideoWriter(
                    "output.mp4",
                    fourcc,
                    20.0,
                    (frame.shape[1], frame.shape[0])
                )

        # If recording, write frames
        if recording and out:
            out.write(frame)

            # Check if 3 seconds passed since last detection
            if time.time() - last_detection_time >= 3:
                print("🛑 No detections for 3s. Stopping recording...")
                recording = False
                out.release()
                out = None
                break  # stop after saving once (remove if you want continuous)

        # Show annotated video
        annotated_frame = results[0].plot()
        cv2.imshow("Webcam", annotated_frame)

        # Quit manually
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

    # Uncomment below to send to ThingsBoard
    # DEVICE_TOKEN = "your_thingsboard_device_token"
    # TB_URL = "http://localhost:8080"
    # send_to_thingsboard(total_distribution, DEVICE_TOKEN, TB_URL)


if __name__ == "__main__":
    main()
