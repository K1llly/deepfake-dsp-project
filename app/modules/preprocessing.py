import os
import cv2
import numpy as np
import mediapipe as mp
import librosa

from config import DEFAULT_SAMPLE_RATE, MODELS_DIR


class PreprocessingModule:
    """Handles face detection/alignment and audio preprocessing."""

    def __init__(self):
        landmarker_model = os.path.join(MODELS_DIR, "face_landmarker.task")
        detector_model = os.path.join(MODELS_DIR, "face_detector.tflite")

        landmarker_options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=landmarker_model),
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
        )
        self.face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
            landmarker_options
        )

        detector_options = mp.tasks.vision.FaceDetectorOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=detector_model),
            min_detection_confidence=0.5,
        )
        self.face_detector = mp.tasks.vision.FaceDetector.create_from_options(
            detector_options
        )

    def detect_face_landmarks(self, image):
        """Detect facial landmarks using MediaPipe FaceLandmarker."""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        results = self.face_landmarker.detect(mp_image)

        if not results.face_landmarks:
            return None

        height, width = image.shape[:2]
        points = []
        for landmark in results.face_landmarks[0]:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            points.append((x, y))

        return np.array(points, dtype=np.int32)

    def detect_face_bounding_box(self, image):
        """Detect face and return bounding box coordinates."""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        results = self.face_detector.detect(mp_image)

        if not results.detections:
            return None

        detection = results.detections[0]
        bounding_box = detection.bounding_box
        height, width = image.shape[:2]

        x = bounding_box.origin_x
        y = bounding_box.origin_y
        w = bounding_box.width
        h = bounding_box.height

        return (x, y, w, h)

    def align_face(self, image, landmarks):
        """Align face based on eye positions for consistent processing."""
        left_eye_indices = [33, 133]
        right_eye_indices = [362, 263]

        left_eye = np.mean(landmarks[left_eye_indices], axis=0)
        right_eye = np.mean(landmarks[right_eye_indices], axis=0)

        delta_x = right_eye[0] - left_eye[0]
        delta_y = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(delta_y, delta_x))

        center = tuple(np.mean([left_eye, right_eye], axis=0).astype(int))
        height, width = image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned_image = cv2.warpAffine(image, rotation_matrix, (width, height))

        return aligned_image

    def normalize_audio(self, audio_data):
        """Normalize audio signal to [-1, 1] range."""
        peak = np.max(np.abs(audio_data))
        if peak == 0:
            return audio_data
        return audio_data / peak

    def resample_audio(self, audio_data, original_rate, target_rate=DEFAULT_SAMPLE_RATE):
        """Resample audio to target sample rate."""
        if original_rate == target_rate:
            return audio_data
        return librosa.resample(
            audio_data, orig_sr=original_rate, target_sr=target_rate
        )

    def preprocess_audio(self, audio_data, original_rate):
        """Full audio preprocessing pipeline: resample + normalize."""
        resampled = self.resample_audio(audio_data, original_rate)
        normalized = self.normalize_audio(resampled)
        return normalized

    def extract_face_region(self, image, landmarks, padding=20):
        """Extract the face region from an image using landmarks."""
        x_min = max(0, np.min(landmarks[:, 0]) - padding)
        y_min = max(0, np.min(landmarks[:, 1]) - padding)
        x_max = min(image.shape[1], np.max(landmarks[:, 0]) + padding)
        y_max = min(image.shape[0], np.max(landmarks[:, 1]) + padding)

        return image[y_min:y_max, x_min:x_max]

    def close(self):
        """Release MediaPipe resources."""
        self.face_landmarker.close()
        self.face_detector.close()
