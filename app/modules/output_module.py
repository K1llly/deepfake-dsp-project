import cv2
import numpy as np
import soundfile as sf

from app.utils.file_utils import generate_output_path


class OutputModule:
    """Handles saving and exporting processed audio/video outputs."""

    def save_audio(self, audio_data, sample_rate, filename="output_audio.wav"):
        """Save audio data to a WAV file."""
        output_path = generate_output_path(filename)
        sf.write(output_path, audio_data, sample_rate)
        return output_path

    def save_image(self, image, filename="output_image.png"):
        """Save a processed image to file."""
        output_path = generate_output_path(filename)
        cv2.imwrite(output_path, image)
        return output_path

    def save_video(self, frames, fps, filename="output_video.mp4"):
        """Save processed video frames to an MP4 file."""
        if not frames:
            return None

        output_path = generate_output_path(filename)
        height, width = frames[0].shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in frames:
            writer.write(frame)

        writer.release()
        return output_path

    def save_video_with_audio(self, frames, fps, audio_path, filename="output_combined.mp4"):
        """Save video frames and combine with audio track."""
        video_path = self.save_video(frames, fps, "_temp_video.mp4")
        if video_path is None:
            return None

        output_path = generate_output_path(filename)

        try:
            import subprocess
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", video_path,
                    "-i", audio_path,
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-shortest",
                    output_path,
                ],
                capture_output=True,
                check=True,
            )
            return output_path
        except (subprocess.CalledProcessError, FileNotFoundError):
            return video_path
