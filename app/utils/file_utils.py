import os
from config import (
    SUPPORTED_IMAGE_FORMATS,
    SUPPORTED_AUDIO_FORMATS,
    SUPPORTED_VIDEO_FORMATS,
    OUTPUTS_DIR,
)


def get_file_extension(file_path):
    _, extension = os.path.splitext(file_path)
    return extension.lower()


def is_supported_image(file_path):
    return get_file_extension(file_path) in SUPPORTED_IMAGE_FORMATS


def is_supported_audio(file_path):
    return get_file_extension(file_path) in SUPPORTED_AUDIO_FORMATS


def is_supported_video(file_path):
    return get_file_extension(file_path) in SUPPORTED_VIDEO_FORMATS


def is_supported_file(file_path):
    return (
        is_supported_image(file_path)
        or is_supported_audio(file_path)
        or is_supported_video(file_path)
    )


def file_exists(file_path):
    return os.path.isfile(file_path)


def ensure_output_directory():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)


def generate_output_path(filename):
    ensure_output_directory()
    return os.path.join(OUTPUTS_DIR, filename)
