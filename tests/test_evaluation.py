"""Unit tests for EvaluationModule.

Each metric is exercised on synthetic signals where the expected behavior
is unambiguous (pure tone, white noise, identical inputs, octave shifts).
The thresholds here are loose on purpose — the goal is to catch breakage,
not to lock in specific numerical values that may drift slightly across
librosa versions.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.modules.evaluation import EvaluationModule


SR = 22050


@pytest.fixture
def ev():
    return EvaluationModule()


def _sine(freq_hz, dur_s=2.0, sr=SR, amp=0.5):
    t = np.linspace(0, dur_s, int(dur_s * sr), endpoint=False)
    return amp * np.sin(2 * np.pi * freq_hz * t)


def _noise(dur_s=2.0, sr=SR, amp=0.3, seed=0):
    rng = np.random.default_rng(seed)
    return amp * rng.standard_normal(int(dur_s * sr))


# ── HNR ──

class TestHNR:
    def test_pure_tone_is_high(self, ev):
        # A pure sine should be perfectly periodic → HNR clamps near +30 dB.
        hnr = ev.calculate_hnr(_sine(150), SR)
        assert hnr > 25.0

    def test_pure_noise_is_low(self, ev):
        # White noise has no periodicity → HNR should be near 0 or negative.
        hnr = ev.calculate_hnr(_noise(seed=1), SR)
        assert hnr < 5.0

    def test_clean_beats_noisy(self, ev):
        clean = ev.calculate_hnr(_sine(150), SR)
        noisy = ev.calculate_hnr(_sine(150) + 0.3 * _noise(seed=2), SR)
        assert clean > noisy

    def test_short_audio_returns_zero(self, ev):
        # Anything below the frame length should bail out instead of crashing.
        assert ev.calculate_hnr(np.zeros(100), SR) == 0.0


# ── Clarity ──

class TestClarity:
    def test_silence_is_zero(self, ev):
        assert ev.calculate_audio_clarity(np.zeros(SR)) == 0.0

    def test_loud_is_high(self, ev):
        # 0.15 RMS is the saturation point in the metric.
        loud = _sine(220, amp=0.5)  # RMS ≈ 0.354
        assert ev.calculate_audio_clarity(loud) >= 0.99

    def test_quiet_is_low(self, ev):
        quiet = _sine(220, amp=0.02)  # RMS ≈ 0.014
        assert ev.calculate_audio_clarity(quiet) < 0.2

    def test_empty_input(self, ev):
        assert ev.calculate_audio_clarity(np.array([])) == 0.0


# ── Voice similarity ──

class TestVoiceSimilarity:
    def test_identical_audio_high(self, ev):
        # Same tone twice should have very high similarity.
        a = _sine(180, dur_s=3.0)
        sim = ev.calculate_voice_similarity(a, SR, a, SR)
        assert sim > 0.95

    def test_unrelated_audio_lower(self, ev):
        a = _sine(180, dur_s=3.0)
        b = _noise(dur_s=3.0, seed=3)
        sim_same = ev.calculate_voice_similarity(a, SR, a, SR)
        sim_diff = ev.calculate_voice_similarity(a, SR, b, SR)
        assert sim_same > sim_diff

    def test_too_short_returns_default(self, ev):
        assert ev.calculate_voice_similarity(
            np.zeros(500), SR, np.zeros(500), SR
        ) == 0.5


# ── Pitch difference (semitones) ──

class TestPitchDifference:
    def test_same_pitch_is_zero(self, ev):
        a = _sine(220, dur_s=2.0)
        diff = ev.calculate_pitch_difference_semitones(a, SR, a, SR)
        assert diff < 0.5  # within half a semitone

    def test_octave_is_twelve(self, ev):
        a = _sine(220, dur_s=2.0)
        b = _sine(440, dur_s=2.0)
        diff = ev.calculate_pitch_difference_semitones(a, SR, b, SR)
        assert 11.0 < diff < 13.0

    def test_fifth_is_seven(self, ev):
        # A perfect fifth (3:2 ratio) is 7 semitones.
        a = _sine(220, dur_s=2.0)
        b = _sine(330, dur_s=2.0)
        diff = ev.calculate_pitch_difference_semitones(a, SR, b, SR)
        assert 6.0 < diff < 8.0


# ── Aggregate ──

class TestEvaluateClone:
    def test_returns_all_keys(self, ev):
        a = _sine(180, dur_s=3.0)
        result = ev.evaluate_clone(a, SR, a, SR)
        assert set(result.keys()) == {
            "hnr_db", "clarity", "voice_similarity", "pitch_semitones",
        }

    def test_self_similarity_is_high(self, ev):
        a = _sine(180, dur_s=3.0)
        result = ev.evaluate_clone(a, SR, a, SR)
        assert result["voice_similarity"] > 0.95
        assert result["pitch_semitones"] < 0.5


# ── PSNR / SSIM (image metrics) ──

class TestImageMetrics:
    def test_psnr_identical_is_infinite(self, ev):
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        assert ev.calculate_psnr(img, img) == float("inf")

    def test_psnr_finite_when_different(self, ev):
        a = np.full((64, 64, 3), 128, dtype=np.uint8)
        b = np.full((64, 64, 3), 100, dtype=np.uint8)
        assert ev.calculate_psnr(a, b) < float("inf")
        assert ev.calculate_psnr(a, b) > 0

    def test_psnr_shape_mismatch_raises(self, ev):
        a = np.zeros((64, 64, 3), dtype=np.uint8)
        b = np.zeros((32, 32, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            ev.calculate_psnr(a, b)

    def test_ssim_identical_is_one(self, ev):
        img = np.random.RandomState(0).randint(
            0, 256, (64, 64, 3), dtype=np.uint8
        )
        assert ev.calculate_ssim(img, img) == pytest.approx(1.0, abs=1e-3)

    def test_ssim_unrelated_is_lower(self, ev):
        rng = np.random.RandomState(0)
        a = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        b = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        assert ev.calculate_ssim(a, b) < 0.5


# ── SNR ──

class TestSNR:
    def test_identical_is_infinite(self, ev):
        a = _sine(220, dur_s=1.0)
        assert ev.calculate_snr(a, a) == float("inf")

    def test_lower_when_signal_corrupted(self, ev):
        clean = _sine(220, dur_s=1.0)
        noisy = clean + 0.3 * _noise(dur_s=1.0, seed=4)
        assert ev.calculate_snr(clean, noisy) < float("inf")
