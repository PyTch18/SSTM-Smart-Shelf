from __future__ import annotations
import cv2
import os
import time
import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple


# -------------------------------
# Pre/post capture windows
# -------------------------------
PREBUFFER_SECONDS = 10.0      # seconds to keep before an event
POST_SECONDS = 2.0            # seconds to keep after an event

# Camera
CAM_INDEX = 0                 # default webcam index
TARGET_CAPTURE_FPS = 20.0     # used if camera FPS not reported
FOURCC = "mp4v"               # mp4 codec
SHOW_PREVIEW = True
OUTPUT_DIR = "recordings"

# Sensor simulation (grams)
BASE_WEIGHT_G = 1500          # set the original/base weight here (grams)
DECISION_INTERVAL_S = 20.0    # how often to consider a change
CHANGE_PROB = 0.6             # probability to toggle state at each decision
DELTA_MIN_G = 50              # when changing, min deviation from base (g)
DELTA_MAX_G = 500             # when changing, max deviation from base (g)
NOISE_G = 3                   # small random noise per reading (±)

# Event thresholds (grams)
LEAVE_BASE_THRESHOLD_G = 80   # leaving base requires >= this deviation from BASE
RETURN_TOLERANCE_G = 25       # consider "back to base" if within this of BASE
SENSOR_POLL_HZ = 10           # how often we poll the sensor logic


# -------------------------------
# Timed Random Sensor (grams)
# -------------------------------
@dataclass
class TimedRandomWeightSensor:
    base_weight_g: int = BASE_WEIGHT_G
    current_weight_g: int = BASE_WEIGHT_G
    decision_interval_s: float = DECISION_INTERVAL_S
    change_prob: float = CHANGE_PROB
    delta_min_g: int = DELTA_MIN_G
    delta_max_g: int = DELTA_MAX_G
    noise_g: int = NOISE_G
    _next_decision_time: float = 0.0
    _last_decision_note: str = "init"

    def _schedule_next(self) -> None:
        now = time.time()
        self._next_decision_time = now + self.decision_interval_s

    def time_to_next_decision(self) -> int:
        return max(0, int(self._next_decision_time - time.time()))

    def read_weight_g(self) -> int:
        """Return current weight (grams) with small noise.
        Every decision interval, with probability `change_prob`, toggle state.
        """
        now = time.time()
        if self._next_decision_time == 0.0:
            self._schedule_next()
        if now >= self._next_decision_time:
            self._schedule_next()
            if random.random() < self.change_prob:
                # Toggle state
                if self.current_weight_g == self.base_weight_g:
                    delta = random.randint(self.delta_min_g, self.delta_max_g)
                    sign = random.choice([-1, 1])
                    self.current_weight_g = self.base_weight_g + sign * delta
                    self._last_decision_note = f"LEAVE_BASE to {self.current_weight_g}g"
                else:
                    self.current_weight_g = self.base_weight_g
                    self._last_decision_note = "RETURN_TO_BASE"
            else:
                self._last_decision_note = "KEEP_STATE"
        # add small noise (bounded so we stay integer grams)
        noisy = self.current_weight_g + random.randint(-self.noise_g, self.noise_g)
        return int(noisy)

    @property
    def last_decision_note(self):
        return self._last_decision_note


# -------------------------------
# Utilities
# -------------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def estimate_fps_from_buffer(stamps: List[float]) -> float:
    if len(stamps) < 2:
        return TARGET_CAPTURE_FPS
    duration = max(1e-3, stamps[-1] - stamps[0])
    fps_est = len(stamps) / duration
    return float(max(5.0, min(30.0, round(fps_est))))


def write_video(filename: str, frames: List, fps: float) -> None:
    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*FOURCC)  # type: ignore
    vw = cv2.VideoWriter(filename, fourcc, fps, (w, h))
    try:
        for f in frames:
            if f is None:
                continue
            if len(f.shape) == 2:
                f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
            vw.write(f)
    finally:
        vw.release()


# -------------------------------
# Main Logic
# -------------------------------

def main():
    ensure_dir(OUTPUT_DIR)

    # Camera setup
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW) if os.name == 'nt' else cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera at index {CAM_INDEX}")

    cam_fps = cap.get(cv2.CAP_PROP_FPS)
    if not cam_fps or math.isnan(cam_fps) or cam_fps <= 0:
        cam_fps = TARGET_CAPTURE_FPS
    frame_interval = 1.0 / cam_fps

    # Rolling buffer of (timestamp, frame)
    buf: Deque[Tuple[float, any]] = deque()

    # Sensor state
    sensor = TimedRandomWeightSensor(base_weight_g=BASE_WEIGHT_G, change_prob=CHANGE_PROB)
    last_weight_g = sensor.read_weight_g()
    next_sensor_poll = time.time()

    # Event + base-return tracking
    near_base_prev = abs(last_weight_g - BASE_WEIGHT_G) <= RETURN_TOLERANCE_G

    event_active = False
    event_time = 0.0
    write_after = 0.0

    print("Running. Press 'q' to quit.")

    try:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            now = time.time()
            buf.append((now, frame.copy()))

            # --- Dynamic buffer trim ---
            if event_active:
                # keep enough to cover pre + post so we can write the full window
                min_keep_time = now - (PREBUFFER_SECONDS + POST_SECONDS + 1.0)
            else:
                # idle: keep only the prebuffer window (~10s)
                min_keep_time = now - PREBUFFER_SECONDS
            while buf and buf[0][0] < min_keep_time:
                buf.popleft()

            # Poll sensor
            if now >= next_sensor_poll:
                next_sensor_poll = now + (1.0 / SENSOR_POLL_HZ)
                w_g = sensor.read_weight_g()

                # Base proximity state
                near_base_now = abs(w_g - BASE_WEIGHT_G) <= RETURN_TOLERANCE_G

                # Detect transitions for recording triggers
                left_base = (near_base_prev is True) and (abs(w_g - BASE_WEIGHT_G) >= LEAVE_BASE_THRESHOLD_G)
                returned_to_base = (near_base_prev is False) and (near_base_now is True)

                if (left_base or returned_to_base) and not event_active:
                    event_active = True
                    event_time = now
                    write_after = now + POST_SECONDS
                    reason = "LEFT_BASE" if left_base else "RETURNED_TO_BASE"
                    print(f"[EVENT] {reason} at {time.strftime('%H:%M:%S')} | weight={w_g}g (base={BASE_WEIGHT_G}g)")

                near_base_prev = near_base_now
                last_weight_g = w_g

            # If an event is active, wait for post window and then write clip
            if event_active and now >= write_after:
                start_t = event_time - PREBUFFER_SECONDS
                end_t = write_after
                stamps = [ts for ts, _ in buf]
                frames = [fr for ts, fr in buf if start_t <= ts <= end_t]
                win_stamps = [ts for ts in stamps if start_t <= ts <= end_t]
                fps = estimate_fps_from_buffer(win_stamps) if win_stamps else cam_fps
                human_ts = time.strftime('%Y%m%d_%H%M%S')
                out_path = os.path.join(OUTPUT_DIR, f"trigger_{human_ts}.mp4")
                write_video(out_path, frames, fps)
                print(f"[SAVED] {out_path} ({len(frames)} frames @ ~{fps:.1f} fps)")
                event_active = False
                event_time = 0.0
                write_after = 0.0
            # Preview overlay
            if SHOW_PREVIEW:
                overlay = frame.copy()
                status_lines = [
                    f"Weight: {last_weight_g} g (base {BASE_WEIGHT_G} g)",
                    f"Near base: {abs(last_weight_g-BASE_WEIGHT_G) <= RETURN_TOLERANCE_G}",
                    f"Next decision in: {sensor.time_to_next_decision()}s  (Δ every {int(DECISION_INTERVAL_S)}s)",
                    f"Last decision: {sensor.last_decision_note}",
                    f"Buffered: ~{max(0.0, now - buf[0][0]):.1f}s / {PREBUFFER_SECONDS:.0f}s",
                    ("EVENT: post-window recording..." if event_active else "Idle: waiting for leave/return"),
                    "Press 'q' to quit",
                ]
                y = 24
                for line in status_lines:
                    cv2.putText(overlay, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                    y += 24
                cv2.imshow("Prebuffer Recorder (grams)", overlay)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            # pace the loop to camera FPS
            elapsed = time.time() - t0
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        cap.release()
        if SHOW_PREVIEW:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
