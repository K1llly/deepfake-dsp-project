import os
import sys
import json
import asyncio
import subprocess
import numpy as np
import librosa
import soundfile as sf
import edge_tts

from config import DEFAULT_SAMPLE_RATE
from app.utils.file_utils import generate_output_path


PRESET_VOICE_PROFILES = {
    "Deep Male": {"voice": "en-US-ChristopherNeural", "pitch_shift": -3},
    "Warm Female": {"voice": "en-US-JennyNeural", "pitch_shift": 0},
    "Very Deep Male": {"voice": "en-US-EricNeural", "pitch_shift": -5},
    "British Female": {"voice": "en-GB-SoniaNeural", "pitch_shift": 0},
    "Casual Male": {"voice": "en-US-GuyNeural", "pitch_shift": -2},
    "British Male": {"voice": "en-GB-RyanNeural", "pitch_shift": 0},
    "Bright Female": {"voice": "en-US-EmmaNeural", "pitch_shift": 0},
    "Default": {"voice": "en-US-AriaNeural", "pitch_shift": 0},
}

F5_SAMPLE_RATE = 24000


class VoiceCloningModule:
    """Voice cloning using f5-tts-mlx (real cloning) + edge-tts (presets)."""

    def __init__(self):
        self.reference_path = None
        self.reference_text = None

    def get_preset_names(self):
        return list(PRESET_VOICE_PROFILES.keys())

    def synthesize_from_preset(self, text, preset_name="Default", output_filename=None):
        """Generate speech using a neural voice preset (edge-tts)."""
        profile = PRESET_VOICE_PROFILES.get(
            preset_name, PRESET_VOICE_PROFILES["Default"]
        )

        if output_filename is None:
            safe_name = preset_name.replace(" ", "_").lower()
            output_filename = f"voice_{safe_name}.wav"

        raw_path = generate_output_path("_raw_neural.mp3")
        final_path = generate_output_path(output_filename)

        self._run_edge_tts(text, profile["voice"], raw_path)

        audio_data, sample_rate = librosa.load(raw_path, sr=DEFAULT_SAMPLE_RATE)
        pitch_shift = profile.get("pitch_shift", 0)
        if pitch_shift != 0:
            audio_data = librosa.effects.pitch_shift(
                y=audio_data, sr=sample_rate, n_steps=pitch_shift
            )

        self._normalize_and_save(audio_data, sample_rate, final_path)
        self._cleanup_temp_file(raw_path)
        return final_path

    def load_reference_voice(self, audio_path, reference_text=""):
        """Load and prepare a reference voice sample for cloning."""
        # Convert to mono 24kHz WAV for f5-tts
        prepared_path = generate_output_path("_ref_prepared.wav")
        audio, _ = librosa.load(audio_path, sr=F5_SAMPLE_RATE, mono=True)

        # Trim silence and quiet sections from edges
        audio, _ = librosa.effects.trim(audio, top_db=20)

        # Find the loudest 5-second segment (most likely clean speech)
        max_samples = F5_SAMPLE_RATE * 5
        if len(audio) > max_samples:
            # Find the segment with highest RMS energy (most speech)
            best_start = 0
            best_rms = 0
            hop = F5_SAMPLE_RATE
            for start in range(0, len(audio) - max_samples, hop):
                segment = audio[start:start + max_samples]
                rms = float(np.sqrt(np.mean(segment ** 2)))
                if rms > best_rms:
                    best_rms = rms
                    best_start = start
            audio = audio[best_start:best_start + max_samples]

        sf.write(prepared_path, audio, F5_SAMPLE_RATE)

        self.reference_path = prepared_path
        self.reference_text = reference_text

        duration = len(audio) / F5_SAMPLE_RATE
        return duration

    def get_profile_summary(self):
        if self.reference_path is None:
            return "No voice sample loaded."
        audio, _ = librosa.load(self.reference_path, sr=F5_SAMPLE_RATE)
        duration = len(audio) / F5_SAMPLE_RATE
        return f"Duration: {duration:.1f}s | Ready for cloning"

    def synthesize_from_reference(self, text, output_filename=None):
        """Clone the reference voice using f5-tts-mlx in a subprocess."""
        if self.reference_path is None:
            raise ValueError("No reference voice loaded.")

        if output_filename is None:
            output_filename = "cloned_voice.wav"

        final_path = generate_output_path(output_filename)

        word_count = len(text.split())
        duration_sec = max(3.0, word_count * 0.4)

        script = "\n".join([
            "import sys, os",
            "from f5_tts_mlx.generate import generate",
            "try:",
            f"    generate(",
            f"        generation_text={json.dumps(text)},",
            f"        ref_audio_path={json.dumps(self.reference_path)},",
            f"        ref_audio_text={json.dumps(self.reference_text or '')},",
            f"        output_path={json.dumps(final_path)},",
            f"        steps=8, speed=1.0,",
            f"    )",
            f"    if os.path.getsize({json.dumps(final_path)}) < 10000:",
            f"        print('Retrying with forced duration')",
            f"        generate(",
            f"            generation_text={json.dumps(text)},",
            f"            ref_audio_path={json.dumps(self.reference_path)},",
            f"            ref_audio_text={json.dumps(self.reference_text or '')},",
            f"            output_path={json.dumps(final_path)},",
            f"            duration={duration_sec},",
            f"            steps=8, speed=1.0,",
            f"        )",
            f"    print('SUCCESS')",
            "except Exception as e:",
            "    print(f'FAIL: {e}', file=sys.stderr)",
            "    sys.exit(1)",
        ])

        env = os.environ.copy()
        env["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=120, env=env,
        )

        if result.returncode != 0:
            error_lines = [l for l in result.stderr.strip().split('\n') if 'FAIL:' in l]
            error_msg = '\n'.join(error_lines) if error_lines else result.stderr[-300:]
            raise RuntimeError(f"Voice cloning failed: {error_msg}")

        if not os.path.exists(final_path) or os.path.getsize(final_path) < 5000:
            raise RuntimeError("Clone failed. Try a cleaner audio sample.")

        return final_path

    def _run_edge_tts(self, text, voice, output_path):
        """Generate speech using edge-tts neural voices."""
        async def _generate():
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(output_path)

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_generate())
        finally:
            loop.close()

    def _normalize_and_save(self, audio, sample_rate, output_path):
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.9
        sf.write(output_path, audio, sample_rate)

    def _cleanup_temp_file(self, file_path):
        if os.path.exists(file_path):
            os.remove(file_path)
