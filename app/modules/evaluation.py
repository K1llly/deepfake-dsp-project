import numpy as np
import librosa


class EvaluationModule:
    """Calculates quality metrics: SNR, PSNR, SSIM."""

    def calculate_snr(self, original_audio, processed_audio):
        """Calculate Signal-to-Noise Ratio in dB.

        Only meaningful when processed = original + noise (same content).
        For unrelated audio (e.g. a voice clone vs the reference), use
        calculate_hnr on the output instead.
        """
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

    def calculate_hnr(self, audio, sample_rate, frame_length=2048, hop_length=512):
        """Harmonics-to-Noise Ratio in dB (output-only audio quality).

        Per-frame normalized autocorrelation peak in the voicing range
        (50-500 Hz fundamental), averaged over voiced frames.
        Clean speech: >20 dB. Noisy/distorted: <10 dB.
        """
        audio, _ = librosa.effects.trim(audio, top_db=25)
        if len(audio) < frame_length:
            return 0.0

        min_lag = max(1, int(sample_rate / 500))
        max_lag = min(int(sample_rate / 50), frame_length - 1)
        if max_lag <= min_lag:
            return 0.0

        frames = librosa.util.frame(
            audio, frame_length=frame_length, hop_length=hop_length
        )

        # Unbiased autocorrelation correction: np.correlate produces a
        # triangular taper because each lag has fewer overlapping samples.
        # Divide by (N - k) per lag to remove that bias before normalizing.
        taper = np.arange(frame_length, 0, -1, dtype=np.float64)

        hnr_values = []
        for i in range(frames.shape[1]):
            frame = frames[:, i]
            rms = float(np.sqrt(np.mean(frame ** 2)))
            if rms < 0.01:
                continue
            ac = np.correlate(frame, frame, mode="full")[frame_length - 1:]
            ac = ac / taper
            if ac[0] <= 0:
                continue
            ac = ac / ac[0]
            peak = float(np.max(ac[min_lag:max_lag]))
            peak = min(0.999, max(0.001, peak))
            hnr_values.append(10 * np.log10(peak / (1 - peak)))

        if not hnr_values:
            return 0.0
        return round(float(np.mean(hnr_values)), 2)

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

    def calculate_audio_clarity(self, audio):
        """RMS-based loudness/clarity score in [0, 1]. 0.15 RMS ≈ full."""
        if len(audio) == 0:
            return 0.0
        rms = float(np.sqrt(np.mean(audio ** 2)))
        return float(min(1.0, rms / 0.15))

    def calculate_voice_similarity(self, ref_audio, ref_sr, out_audio, out_sr):
        """Cosine similarity over an MFCC-based voice embedding.

        Skips MFCC[0] (log-energy), augments with delta and delta-delta
        means, L2-normalizes, then takes cosine. Trims silence on both
        clips so the embedding reflects voiced speech only. Returns a
        value in [0, 1] — same speaker should land 0.80-0.95.
        """
        min_len = min(len(ref_audio), len(out_audio))
        if min_len < 1000:
            return 0.5
        ref_voiced, _ = librosa.effects.trim(ref_audio, top_db=25)
        out_voiced, _ = librosa.effects.trim(out_audio, top_db=25)
        ref_feat = self._voice_embed(ref_voiced, ref_sr)
        out_feat = self._voice_embed(out_voiced, out_sr)
        cosine_sim = float(np.dot(ref_feat, out_feat))
        return float(max(0.0, min(1.0, cosine_sim)))

    def _voice_embed(self, y, sr):
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[1:]
        d1 = librosa.feature.delta(mfcc)
        d2 = librosa.feature.delta(mfcc, order=2)
        feat = np.concatenate([
            np.mean(mfcc, axis=1),
            np.mean(d1, axis=1),
            np.mean(d2, axis=1),
        ])
        return feat / (np.linalg.norm(feat) + 1e-8)

    def calculate_pitch_difference_semitones(
        self, ref_audio, ref_sr, out_audio, out_sr
    ):
        """Median-pitch difference in semitones (musical interval).

        Robust to register: a 50 Hz raw difference at 100 Hz vs 300 Hz
        means very different perceptual changes, but one semitone is one
        semitone everywhere. Returns absolute interval in semitones.
        """
        ref_f0, _, _ = librosa.pyin(ref_audio, fmin=50, fmax=500, sr=ref_sr)
        out_f0, _, _ = librosa.pyin(out_audio, fmin=50, fmax=500, sr=out_sr)
        ref_clean = ref_f0[~np.isnan(ref_f0)] if ref_f0 is not None else np.array([])
        out_clean = out_f0[~np.isnan(out_f0)] if out_f0 is not None else np.array([])
        ref_median = float(np.median(ref_clean)) if len(ref_clean) > 0 else 130.0
        out_median = float(np.median(out_clean)) if len(out_clean) > 0 else 130.0
        return float(abs(12.0 * np.log2(out_median / max(ref_median, 1e-3))))

    def evaluate_clone(self, ref_audio, ref_sr, out_audio, out_sr):
        """Aggregate metrics for a voice clone vs its reference.

        Returns a dict the UI can render directly. Each metric is also
        callable individually for testing.
        """
        return {
            "hnr_db": self.calculate_hnr(out_audio, out_sr),
            "clarity": self.calculate_audio_clarity(out_audio),
            "voice_similarity": self.calculate_voice_similarity(
                ref_audio, ref_sr, out_audio, out_sr
            ),
            "pitch_semitones": self.calculate_pitch_difference_semitones(
                ref_audio, ref_sr, out_audio, out_sr
            ),
        }

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
