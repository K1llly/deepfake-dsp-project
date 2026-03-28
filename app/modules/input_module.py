import os
import cv2
import numpy as np
import librosa
import soundfile as sf

from app.utils.file_utils import (
    file_exists,
    is_supported_image,
    is_supported_audio,
    is_supported_video,
    get_file_extension,
)
from config import DEFAULT_SAMPLE_RATE


class InputValidationError(Exception):
    pass


class InputModule:
    """Handles loading and validating all user inputs."""

    def load_image(self, file_path):
        self._validate_file(file_path, is_supported_image, "image")
        image = cv2.imread(file_path)
        if image is None:
            raise InputValidationError(f"Failed to read image: {file_path}")
        return image

    def load_audio(self, file_path, target_sample_rate=DEFAULT_SAMPLE_RATE):
        self._validate_file(file_path, is_supported_audio, "audio")
        audio_data, sample_rate = librosa.load(
            file_path, sr=target_sample_rate
        )
        return audio_data, sample_rate

    def load_video(self, file_path):
        self._validate_file(file_path, is_supported_video, "video")
        capture = cv2.VideoCapture(file_path)
        if not capture.isOpened():
            raise InputValidationError(f"Failed to open video: {file_path}")
        return capture

    def load_video_frames(self, file_path):
        capture = self.load_video(file_path)
        frames = []
        while True:
            success, frame = capture.read()
            if not success:
                break
            frames.append(frame)
        capture.release()
        return frames

    def load_text(self, text):
        if not text or not text.strip():
            raise InputValidationError("Text input cannot be empty.")
        return text.strip()

    def get_video_properties(self, file_path):
        capture = self.load_video(file_path)
        properties = {
            "fps": capture.get(cv2.CAP_PROP_FPS),
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
        capture.release()
        return properties

    def _validate_file(self, file_path, format_checker, file_type):
        if not file_exists(file_path):
            raise InputValidationError(f"File not found: {file_path}")
        if not format_checker(file_path):
            extension = get_file_extension(file_path)
            raise InputValidationError(
                f"Unsupported {file_type} format: {extension}"
            )
