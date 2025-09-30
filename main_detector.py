import time
from ultralytics import YOLO  # Updated import
import cv2
from collections import defaultdict
import requests
import json
import base64
import boto3
from datetime import datetime
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import os, pickle

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
# --- Cloudflare R2 credentials ---
ACCOUNT_ID = "e211aa6b51fd20ac35a5db3d56ecc2b5"
WORKER_URL = "https://moneysstm.amr2018azouz.workers.dev/"
ACCESS_KEY = "5d4c8643e1bcd7a1b45df55d7f72499f"
SECRET_KEY = "fd28ed018c041c7c3d4b4267bcea1dab35705cf1f412936ef768362ff6c30c5d"
BUCKET_NAME = "moneyrec"          # the bucket name
ENDPOINT_URL = f"https://e211aa6b51fd20ac35a5db3d56ecc2b5.r2.cloudflarestorage.com"

# Create a client once
s3 = boto3.client(
    's3',
    endpoint_url=ENDPOINT_URL,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY
)

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

        #Only fatal errors will appear, if you want to see debug messages, comment the following line
        cv2.setLogLevel(1)

        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"✅ Camera found at index {i}")
            available.append(i)
            cap.release()
    return available

#def upload_video_to_thingsboard(video_path, token, tb_url="http://localhost:8080"):
    """Upload a video to ThingsBoard with a progress bar and delete after upload."""
    file_size = os.path.getsize(video_path)

    # stream read file in chunks to show progress bar
    def file_stream():
        with open(video_path, "rb") as f, tqdm(
            total=file_size, unit="B", unit_scale=True, desc=f"Uploading {os.path.basename(video_path)}"
        ) as pbar:
            while True:
                chunk = f.read(1024 * 64)
                if not chunk:
                    break
                pbar.update(len(chunk))
                yield chunk

    video_bytes = open(video_path, "rb").read()
    video_b64 = base64.b64encode(video_bytes).decode("utf-8")
    payload = {"video_data": video_b64}

    url = f"{tb_url}/api/v1/{token}/telemetry"
    headers = {"Content-Type": "application/json"}

    # Show progress bar for sending
    with tqdm(total=len(json.dumps(payload)), unit="B", unit_scale=True, desc="Sending JSON") as pbar:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        pbar.update(len(json.dumps(payload)))

    if response.status_code == 200:
        print(f"[SUCCESS] {video_path} uploaded to ThingsBoard.")
        os.remove(video_path)  # ✅ delete after upload
        print(f"[INFO] Deleted local file {video_path}")
    else:
        print(f"[ERROR] Upload failed. Status code: {response.status_code}")
        print("Response:", response.text)

def upload_video_to_r2(video_path):
    # name file with datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{timestamp}.mp4"

    # upload to R2
    s3.upload_file(video_path, BUCKET_NAME, filename, ExtraArgs={"ContentType": "video/mp4"})

    # return public Worker URL
    return f"{WORKER_URL}video/{filename}"


def upload_video_to_r2_2(video_path):
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ext = os.path.splitext(video_path)[1]
    filename = f"{now}{ext}"

    s3.upload_file(
        video_path,
        BUCKET_NAME,
        filename,
        ExtraArgs={
            "ContentType": "video/mp4",  # so browsers know it’s a video
            "ACL": "public-read",  # make it accessible if bucket is public
            "Access-Control-Allow-Origin": "*"
    }
    )

    # Construct your public URL:
    # If you’ve bound a custom domain to the bucket:
    # public_url = f"https://videos.yourdomain.com/{filename}"
    # Otherwise use Cloudflare’s pub-xxx.r2.dev URL (from the dashboard):
    public_url = f"https://pub-aef7cfb1eb824b2693647f855caf80f4.r2.dev/{filename}"

    print(f"[SUCCESS] {video_path} uploaded to R2 as {filename}")
    print(f"[INFO] Public URL: {public_url}")
    return public_url

def main():
    model = YOLO("detection_model.pt")

    cams = list_available_cameras()
    if not cams:
        print("Error: No camera found.")
        return

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
    last_detection_time = 0
    video_counter = 1  # ✅ to increment filenames

    DEVICE_TOKEN = "dm1weaxrbl4a3lvji5ja"
    TB_URL = "https://demo.thingsboard.io"

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame.")
            break

        # Run YOLO detection
        results = model(frame, imgsz=720, conf=0.1)
        detections = results[0].boxes

        if len(detections) > 0:
            last_detection_time = time.time()
            if not recording:
                output_path = f"video_{video_counter}.mp4"  # ✅ incremented filename
                print(f"🎥 Detection found! Start recording to {output_path}...")
                recording = True
                out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame.shape[1], frame.shape[0]))

        if recording and out:
            out.write(frame)

            # If no detections for ≥3 seconds, stop recording and upload
            if time.time() - last_detection_time >= 3:
                print("🛑 No detections for 3s. Stopping recording...")
                recording = False
                out.release()
                out = None

                # ✅ Upload to ThingsBoard with progress bar
                public_url = upload_video_to_r2(output_path)
                print("Public URL for ThingsBoard:", public_url)

                # (Optional) send the URL itself to ThingsBoard so it can display the video link:
                payload = {"video_url": public_url}
                url = f"{TB_URL}/api/v1/{DEVICE_TOKEN}/telemetry"
                headers = {"Content-Type": "application/json"}
                requests.post(url, headers=headers, data=json.dumps(payload))

                # ✅ increment counter for next file
                video_counter += 1

        annotated_frame = results[0].plot()
        cv2.imshow("Webcam", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

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