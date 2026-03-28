import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

SUPPORTED_IMAGE_FORMATS = (".jpg", ".jpeg", ".png")
SUPPORTED_AUDIO_FORMATS = (".wav", ".mp3")
SUPPORTED_VIDEO_FORMATS = (".mp4", ".avi", ".mov")

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_AUDIO_FORMAT = "wav"

WINDOW_TITLE = "DeepFake Interaction System"
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 750

APPEARANCE_MODE = "dark"
COLOR_THEME = "blue"
