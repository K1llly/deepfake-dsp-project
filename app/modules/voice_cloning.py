import os
import asyncio
import threading
import numpy as np
import librosa
import soundfile as sf
import edge_tts

from config import DEFAULT_SAMPLE_RATE
from app.utils.file_utils import generate_output_path

# Auto-accept the Coqui XTTS-v2 license terms so the first call doesn't
# block on an interactive prompt.
os.environ.setdefault("COQUI_TOS_AGREED", "1")


PRESET_VOICE_PROFILES = {
    "Deep Male": {"voice": "en-US-ChristopherNeural", "pitch_shift": -3},
    "Warm Female": {"voice": "en-US-JennyNeural", "pitch_shift": 0},
    "Very Deep Male": {"voice": "en-US-EricNeural", "pitch_shift": -5},
    "British Female": {"voice": "en-GB-SoniaNeural", "pitch_shift": 0},
    "Casual Male": {"voice": "en-US-GuyNeural", "pitch_shift": -2},
    "British Male": {"voice": "en-GB-RyanNeural", "pitch_shift": 0},
    "Bright Female": {"voice": "en-US-EmmaNeural", "pitch_shift": 0},
    "Default": {"voice": "en-US-AriaNeural", "pitch_shift": 0},
    "Turkish Male (Ahmet)": {"voice": "tr-TR-AhmetNeural", "pitch_shift": 0},
    "Turkish Female (Emel)": {"voice": "tr-TR-EmelNeural", "pitch_shift": 0},
}

# XTTS-v2 supported language codes (ISO 639-1, with zh-cn for Mandarin).
XTTS_LANGUAGES = {
    "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru",
    "nl", "cs", "ar", "zh-cn", "hu", "ko", "ja", "hi",
}


class VoiceCloningModule:
    """Voice cloning via Coqui XTTS-v2 (multilingual, cross-lingual) and
    edge-tts neural voices for the preset path.
    """

    def __init__(self):
        self.reference_path = None
        self.reference_text = None
        self._xtts = None
        self._xtts_load_lock = threading.Lock()
        # Background-load the XTTS model so the user's first clone doesn't
        # pay the ~10 s cold-start. Subsequent calls reuse the loaded model.
        threading.Thread(target=self._load_xtts, daemon=True).start()

    def _load_xtts(self):
        """Idempotently load XTTS-v2; safe to call from multiple threads."""
        with self._xtts_load_lock:
            if self._xtts is not None:
                return self._xtts
            from TTS.api import TTS
            self._xtts = TTS(
                model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                progress_bar=False,
            )
            return self._xtts

    def is_ready(self):
        """True once the XTTS model has finished loading in the background."""
        return self._xtts is not None

    def get_preset_names(self):
        return list(PRESET_VOICE_PROFILES.keys())

    def synthesize_from_preset(self, text, preset_name="Default", output_filename=None):
        """Generate speech using a neural voice preset (edge-tts)."""
        profile = PRESET_VOICE_PROFILES.get(
            preset_name, PRESET_VOICE_PROFILES["Default"]
        )
        if output_filename is None:
            safe = (
                preset_name.replace(" ", "_")
                .replace("(", "").replace(")", "").lower()
            )
            output_filename = f"voice_{safe}.wav"

        raw_path = generate_output_path("_raw_neural.mp3")
        final_path = generate_output_path(output_filename)

        self._run_edge_tts(text, profile["voice"], raw_path)

        audio_data, sample_rate = librosa.load(raw_path, sr=DEFAULT_SAMPLE_RATE)
        pitch_shift = profile.get("pitch_shift", 0)
        if pitch_shift != 0:
            audio_data = librosa.effects.pitch_shift(
                y=audio_data, sr=sample_rate, n_steps=pitch_shift,
            )

        self._normalize_and_save(audio_data, sample_rate, final_path)
        self._cleanup_temp_file(raw_path)
        return final_path

    def load_reference_voice(self, audio_path, reference_text=""):
        """Prepare a reference audio sample for cloning. XTTS handles its
        own resampling, so we just trim silence and cap the length at 15 s.
        """
        prepared_path = generate_output_path("_ref_prepared.wav")
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        audio, _ = librosa.effects.trim(audio, top_db=20)
        max_samples = int(15 * sr)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        sf.write(prepared_path, audio, sr)
        self.reference_path = prepared_path
        self.reference_text = reference_text
        return len(audio) / sr

    def get_profile_summary(self):
        if self.reference_path is None:
            return "No voice sample loaded."
        audio, sr = librosa.load(self.reference_path, sr=None)
        return f"Duration: {len(audio)/sr:.1f} s | Ready for cloning"

    def synthesize_from_reference(self, text, output_filename=None, language="tr"):
        """Clone the reference voice using XTTS-v2.

        language must be one of XTTS_LANGUAGES; falls back to English if
        an unsupported code is passed.
        """
        if self.reference_path is None:
            raise ValueError("No reference voice loaded.")
        if language not in XTTS_LANGUAGES:
            language = "en"
        if output_filename is None:
            output_filename = "cloned_voice.wav"
        final_path = generate_output_path(output_filename)

        tts = self._load_xtts()
        try:
            tts.tts_to_file(
                text=text,
                speaker_wav=self.reference_path,
                language=language,
                file_path=final_path,
            )
        except Exception as e:
            raise RuntimeError(f"Voice cloning failed: {e}")

        if not os.path.exists(final_path) or os.path.getsize(final_path) < 5000:
            raise RuntimeError("Clone failed. Try a cleaner audio sample.")
        return final_path

    def _run_edge_tts(self, text, voice, output_path):
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
