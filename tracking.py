# tracking.py — MediaPipe head-tracking daemon thread.

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mptest
from mediapipe.tasks.python import vision
import threading
import time

import config


class HeadTracking:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            print("ERROR: Camera failed to open")
            return

        base_options = mptest.BaseOptions(model_asset_path=config.MODEL_PATH)

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1
        )

        self.landmarker = vision.FaceLandmarker.create_from_options(options)

        self.head_x  = 0.0
        self.head_y  = 0.0
        self.detected = False
        self.running  = True
        self.timestamp = 0

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            success, frame = self.cap.read()
            if not success:
                time.sleep(0.05)
                continue

            # frame = cv2.flip(frame, 1)  # uncomment to mirror camera
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            result = self.landmarker.detect_for_video(mp_image, self.timestamp)
            self.timestamp += 1

            if result.face_landmarks:
                self.detected = True
                pt = result.face_landmarks[0][168]  # nose bridge / centre
                self.head_x = (pt.x - 0.5) * 2
                self.head_y = (pt.y - 0.5) * 2
            else:
                self.detected = False

            time.sleep(1 / 60)

    def stop(self):
        self.running = False
        self.cap.release()
