import numpy as np
import librosa


class EvaluationModule:
    """Calculates quality metrics: SNR, PSNR, SSIM."""

    def calculate_snr(self, original_audio, processed_audio):
        """Calculate Signal-to-Noise Ratio in dB."""
        min_length = min(len(original_audio), len(processed_audio))
        original = original_audio[:min_length]
        processed = processed_audio[:min_length]

        noise = original - processed
        signal_power = np.sum(original ** 2)
        noise_power = np.sum(noise ** 2)

        if noise_power == 0:
            return float("inf")

        snr = 10 * np.log10(signal_power / noise_power)
        return round(snr, 2)

    def calculate_psnr(self, original_image, processed_image):
        """Calculate Peak Signal-to-Noise Ratio in dB."""
        if original_image.shape != processed_image.shape:
            raise ValueError("Images must have the same dimensions for PSNR.")

        original = original_image.astype(np.float64)
        processed = processed_image.astype(np.float64)

        mse = np.mean((original - processed) ** 2)
        if mse == 0:
            return float("inf")

        max_pixel_value = 255.0
        psnr = 10 * np.log10((max_pixel_value ** 2) / mse)
        return round(psnr, 2)

    def calculate_ssim(self, original_image, processed_image):
        """Calculate Structural Similarity Index."""
        if original_image.shape != processed_image.shape:
            raise ValueError("Images must have the same dimensions for SSIM.")

        original = original_image.astype(np.float64)
        processed = processed_image.astype(np.float64)

        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2

        mean_original = np.mean(original)
        mean_processed = np.mean(processed)
        variance_original = np.var(original)
        variance_processed = np.var(processed)
        covariance = np.mean(
            (original - mean_original) * (processed - mean_processed)
        )

        numerator = (
            (2 * mean_original * mean_processed + c1)
            * (2 * covariance + c2)
        )
        denominator = (
            (mean_original ** 2 + mean_processed ** 2 + c1)
            * (variance_original + variance_processed + c2)
        )

        ssim = numerator / denominator
        return round(ssim, 4)

    def evaluate_audio(self, original_audio, processed_audio):
        """Run full audio evaluation."""
        return {
            "snr_db": self.calculate_snr(original_audio, processed_audio),
        }

    def evaluate_image(self, original_image, processed_image):
        """Run full image/video frame evaluation."""
        return {
            "psnr_db": self.calculate_psnr(original_image, processed_image),
            "ssim": self.calculate_ssim(original_image, processed_image),
        }

    def evaluate_video_frames(self, original_frames, processed_frames):
        """Evaluate video quality across frames."""
        frame_count = min(len(original_frames), len(processed_frames))
        psnr_values = []
        ssim_values = []

        for i in range(frame_count):
            psnr = self.calculate_psnr(original_frames[i], processed_frames[i])
            ssim = self.calculate_ssim(original_frames[i], processed_frames[i])
            psnr_values.append(psnr)
            ssim_values.append(ssim)

        return {
            "average_psnr_db": round(np.mean(psnr_values), 2),
            "average_ssim": round(np.mean(ssim_values), 4),
            "frame_count": frame_count,
        }
