import os
import io
import threading
import numpy as np
import customtkinter as ctk
from tkinter import filedialog, Canvas
import sounddevice as sd
import soundfile as sf
import librosa
import noisereduce as nr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from app.modules.voice_cloning import VoiceCloningModule
from app.modules.evaluation import EvaluationModule
from app.utils.logger import get_logger
from config import DEFAULT_SAMPLE_RATE


class VoicePanel(ctk.CTkFrame):
    """UI panel for voice cloning with waveform timeline selector."""

    def __init__(self, parent):
        super().__init__(parent)

        self.voice_module = VoiceCloningModule()
        self.evaluation = EvaluationModule()
        self.log = get_logger()
        self.use_reference = False

        # Raw audio data for timeline
        self.raw_audio = None
        self.raw_sr = None
        self.raw_duration = 0
        self.raw_path = None
        self.sel_start = 0.0
        self.sel_end = 5.0

        self._build_layout()

    def _build_layout(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Scrollable main frame
        main = ctk.CTkScrollableFrame(self)
        main.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        main.grid_columnconfigure(0, weight=1)
        self.main = main

        ctk.CTkLabel(
            main, text="Voice Cloning", font=("Arial", 20, "bold")
        ).pack(pady=(10, 8))

        # --- Voice Source Tabs ---
        source_tabs = ctk.CTkTabview(main, height=150)
        source_tabs.pack(pady=5, padx=15, fill="x")
        self.source_tabs = source_tabs

        # Preset tab
        preset_tab = source_tabs.add("Preset Voices")
        preset_tab.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            preset_tab, text="Select a voice style:", font=("Arial", 12)
        ).grid(row=0, column=0, pady=(5, 0), sticky="w")

        self.preset_selector = ctk.CTkComboBox(
            preset_tab, values=self.voice_module.get_preset_names(),
            state="readonly"
        )
        self.preset_selector.grid(row=1, column=0, pady=8, sticky="ew")
        self.preset_selector.set("Deep Male")

        # Upload tab
        upload_tab = source_tabs.add("Upload Sample")
        upload_tab.grid_columnconfigure(0, weight=1)

        self.upload_button = ctk.CTkButton(
            upload_tab, text="Upload Voice Sample (MP3/WAV)",
            command=self._on_upload_voice_sample
        )
        self.upload_button.grid(row=0, column=0, pady=(5, 5), sticky="ew")

        self.profile_label = ctk.CTkLabel(
            upload_tab, text="Upload clean speech audio (no music/background noise)",
            text_color="gray", font=("Arial", 11), wraplength=400
        )
        self.profile_label.grid(row=1, column=0, pady=(0, 2), sticky="w")


        # --- Audio Selector (hidden until audio loaded) ---
        self.selector_frame = ctk.CTkFrame(main)
        self.selector_visible = False

        # --- Text Input ---
        ctk.CTkLabel(
            main, text="Text to Speak:", font=("Arial", 13)
        ).pack(pady=(10, 0), padx=15, anchor="w")

        self.text_input = ctk.CTkTextbox(main, height=80)
        self.text_input.pack(pady=5, padx=15, fill="x")

        # --- Adjustments ---
        ctk.CTkLabel(
            main, text="Adjustments", font=("Arial", 14, "bold")
        ).pack(pady=(8, 3), padx=15, anchor="w")

        pitch_row = ctk.CTkFrame(main, fg_color="transparent")
        pitch_row.pack(pady=2, padx=15, fill="x")
        ctk.CTkLabel(pitch_row, text="Pitch:", width=55).pack(side="left")
        self.pitch_slider = ctk.CTkSlider(pitch_row, from_=-10, to=10, number_of_steps=20)
        self.pitch_slider.pack(side="left", expand=True, fill="x", padx=5)
        self.pitch_slider.set(0)
        self.pitch_val = ctk.CTkLabel(pitch_row, text="0", width=30)
        self.pitch_val.pack(side="left")
        self.pitch_slider.configure(command=lambda v: self.pitch_val.configure(text=str(int(float(v)))))

        speed_row = ctk.CTkFrame(main, fg_color="transparent")
        speed_row.pack(pady=2, padx=15, fill="x")
        ctk.CTkLabel(speed_row, text="Speed:", width=55).pack(side="left")
        self.speed_slider = ctk.CTkSlider(speed_row, from_=0.5, to=2.0, number_of_steps=15)
        self.speed_slider.pack(side="left", expand=True, fill="x", padx=5)
        self.speed_slider.set(1.0)
        self.speed_val = ctk.CTkLabel(speed_row, text="1.0x", width=40)
        self.speed_val.pack(side="left")
        self.speed_slider.configure(command=lambda v: self.speed_val.configure(text=f"{float(v):.1f}x"))

        # --- Generate & Play ---
        self.generate_button = ctk.CTkButton(
            main, text="Generate & Play", command=self._on_generate,
            height=45, font=("Arial", 15, "bold"),
            fg_color="#2ecc71", hover_color="#27ae60"
        )
        self.generate_button.pack(pady=(15, 3), padx=15, fill="x")

        # Loading bar (hidden by default)
        self.loading_bar = ctk.CTkProgressBar(main, mode="indeterminate")
        self.loading_visible = False

        self.status = ctk.CTkLabel(main, text="", text_color="gray")
        self.status.pack(pady=(3, 3))

        # Replay button (hidden until first generation)
        self.last_output_path = None
        self.replay_button = ctk.CTkButton(
            main, text="Replay Last Output", command=self._on_replay,
            height=35, fg_color="#3498db", hover_color="#2980b9"
        )
        self.replay_visible = False

        # --- Results & Visualizations (shown after generation) ---
        self.results_frame = ctk.CTkFrame(main)
        self.results_visible = False

    # ── Results & Visualizations ──

    def _start_loading(self, message):
        """Show loading bar and status."""
        if not self.loading_visible:
            self.loading_bar.pack(pady=(0, 3), padx=15, fill="x", before=self.status)
            self.loading_visible = True
        self.loading_bar.start()
        self.status.configure(text=message, text_color="yellow")
        self.generate_button.configure(state="disabled")

    def _stop_loading(self):
        """Hide loading bar."""
        self.loading_bar.stop()
        if self.loading_visible:
            self.loading_bar.pack_forget()
            self.loading_visible = False
        self.generate_button.configure(state="normal")

    def _show_replay(self, output_path):
        """Show replay button after generation."""
        self.last_output_path = output_path
        if not self.replay_visible:
            self.replay_button.pack(pady=(0, 5), padx=15, fill="x", before=self.status)
            self.replay_visible = True

    def _on_replay(self):
        """Replay the last generated audio."""
        if self.last_output_path and os.path.exists(self.last_output_path):
            self.status.configure(text="Playing...", text_color="cyan")
            def play():
                self._play_audio(self.last_output_path)
                self.after(0, lambda: self.status.configure(
                    text="Done!", text_color="green"
                ))
            threading.Thread(target=play, daemon=True).start()

    def _show_results(self, ref_audio_path, output_path):
        """Show visual quality gauges, SNR, spectrograms, and waveforms."""
        self.log.info("Generating visualizations...")

        if self.results_visible:
            self.results_frame.destroy()

        self.results_frame = ctk.CTkFrame(self.main)
        self.results_frame.pack(pady=5, padx=15, fill="x")
        self.results_visible = True

        ctk.CTkLabel(
            self.results_frame, text="Output Analysis",
            font=("Arial", 14, "bold")
        ).pack(pady=(8, 5))

        try:
            ref_audio, ref_sr = librosa.load(ref_audio_path, sr=DEFAULT_SAMPLE_RATE)
            out_audio, out_sr = librosa.load(output_path, sr=DEFAULT_SAMPLE_RATE)
            out_duration = len(out_audio) / out_sr
            min_len = min(len(ref_audio), len(out_audio))

            # --- Compute Metrics ---

            # SNR
            if min_len > 1000:
                snr = self.evaluation.calculate_snr(ref_audio[:min_len], out_audio[:min_len])
                snr_val = snr if snr != float("inf") else 50
            else:
                snr_val = 0
            snr_pct = min(1.0, max(0, snr_val / 40))
            snr_label = f"{snr_val:.1f} dB"
            snr_color = "#2ecc71" if snr_val > 20 else "#f39c12" if snr_val > 10 else "#e74c3c"

            # Audio Clarity (RMS energy)
            rms = float(np.sqrt(np.mean(out_audio ** 2)))
            clarity_pct = min(1.0, rms / 0.15)
            clarity_label = "Excellent" if clarity_pct > 0.7 else "Good" if clarity_pct > 0.4 else "Weak"
            clarity_color = "#2ecc71" if clarity_pct > 0.7 else "#f39c12" if clarity_pct > 0.4 else "#e74c3c"

            # Voice Match (MFCC cosine similarity)
            if min_len > 1000:
                ref_mfcc = np.mean(librosa.feature.mfcc(y=ref_audio[:min_len], sr=ref_sr, n_mfcc=13), axis=1)
                out_mfcc = np.mean(librosa.feature.mfcc(y=out_audio[:min_len], sr=out_sr, n_mfcc=13), axis=1)
                cosine_sim = float(np.dot(ref_mfcc, out_mfcc) / (np.linalg.norm(ref_mfcc) * np.linalg.norm(out_mfcc) + 1e-8))
                match_pct = max(0, cosine_sim)
            else:
                match_pct = 0.5
            match_label = "High" if match_pct > 0.85 else "Medium" if match_pct > 0.6 else "Low"
            match_color = "#2ecc71" if match_pct > 0.85 else "#f39c12" if match_pct > 0.6 else "#e74c3c"

            # Pitch Accuracy
            ref_f0, _, _ = librosa.pyin(ref_audio[:min_len], fmin=50, fmax=500, sr=ref_sr)
            out_f0, _, _ = librosa.pyin(out_audio[:min_len], fmin=50, fmax=500, sr=out_sr)
            ref_f0_clean = ref_f0[~np.isnan(ref_f0)] if ref_f0 is not None else np.array([130])
            out_f0_clean = out_f0[~np.isnan(out_f0)] if out_f0 is not None else np.array([130])
            ref_median = float(np.median(ref_f0_clean)) if len(ref_f0_clean) > 0 else 130
            out_median = float(np.median(out_f0_clean)) if len(out_f0_clean) > 0 else 130
            pitch_diff = abs(ref_median - out_median)
            pitch_pct = max(0, 1.0 - pitch_diff / 100)
            pitch_label = f"{pitch_diff:.0f} Hz off"
            pitch_color = "#2ecc71" if pitch_diff < 20 else "#f39c12" if pitch_diff < 50 else "#e74c3c"

            # --- Quality Gauges (4 columns) ---
            gauges = ctk.CTkFrame(self.results_frame)
            gauges.pack(pady=5, padx=10, fill="x")
            for i in range(4):
                gauges.grid_columnconfigure(i, weight=1)

            self._build_gauge(gauges, 0, "SNR", snr_pct, snr_label, snr_color)
            self._build_gauge(gauges, 1, "Audio Clarity", clarity_pct, clarity_label, clarity_color)
            self._build_gauge(gauges, 2, "Voice Match", match_pct, match_label, match_color)
            self._build_gauge(gauges, 3, "Pitch Accuracy", pitch_pct, pitch_label, pitch_color)

            self.log.success(
                f"SNR: {snr_label} | Clarity: {clarity_label} | "
                f"Match: {match_label} ({match_pct:.0%}) | Pitch: {pitch_label}"
            )

            # --- Plots: 2x2 grid ---
            fig, axes = plt.subplots(2, 2, figsize=(10, 5), dpi=85)
            fig.patch.set_facecolor("#1a1a1a")

            for ax in axes.flat:
                ax.set_facecolor("#1a1a1a")
                ax.tick_params(colors="#888888", labelsize=7)
                for spine in ax.spines.values():
                    spine.set_color("#333333")

            # 1. Reference Spectrogram
            ref_spec = librosa.feature.melspectrogram(y=ref_audio, sr=ref_sr, n_mels=80)
            ref_db = librosa.power_to_db(ref_spec, ref=np.max)
            axes[0, 0].imshow(ref_db, aspect="auto", origin="lower", cmap="inferno",
                             extent=[0, len(ref_audio)/ref_sr, 0, ref_sr/2000])
            axes[0, 0].set_title("Reference Spectrogram", color="#3498db", fontsize=9)
            axes[0, 0].set_ylabel("Freq (kHz)", color="#888888", fontsize=7)

            # 2. Output Spectrogram
            out_spec = librosa.feature.melspectrogram(y=out_audio, sr=out_sr, n_mels=80)
            out_db = librosa.power_to_db(out_spec, ref=np.max)
            axes[0, 1].imshow(out_db, aspect="auto", origin="lower", cmap="magma",
                             extent=[0, out_duration, 0, out_sr/2000])
            axes[0, 1].set_title("Cloned Spectrogram", color="#2ecc71", fontsize=9)
            axes[0, 1].set_ylabel("Freq (kHz)", color="#888888", fontsize=7)

            # 3. Waveform Comparison
            ref_time = np.linspace(0, len(ref_audio)/ref_sr, len(ref_audio))
            out_time = np.linspace(0, out_duration, len(out_audio))
            axes[1, 0].plot(ref_time, ref_audio, color="#3498db", linewidth=0.5, alpha=0.8, label="Reference")
            axes[1, 0].plot(out_time, out_audio, color="#2ecc71", linewidth=0.5, alpha=0.8, label="Cloned")
            axes[1, 0].set_title("Waveform Comparison", color="#cccccc", fontsize=9)
            axes[1, 0].set_xlabel("Time (s)", color="#888888", fontsize=7)
            axes[1, 0].legend(fontsize=7, facecolor="#1a1a1a", edgecolor="#333333", labelcolor="#cccccc")

            # 4. Frequency Spectrum Comparison
            if min_len > 100:
                ref_fft = np.abs(np.fft.rfft(ref_audio[:min_len]))
                out_fft = np.abs(np.fft.rfft(out_audio[:min_len]))
                freqs = np.fft.rfftfreq(min_len, 1.0 / DEFAULT_SAMPLE_RATE)
                mask = freqs < 4000
                axes[1, 1].plot(freqs[mask], ref_fft[mask], color="#3498db",
                               alpha=0.7, linewidth=0.8, label="Reference")
                axes[1, 1].plot(freqs[mask], out_fft[mask], color="#2ecc71",
                               alpha=0.7, linewidth=0.8, label="Cloned")
            axes[1, 1].set_title("Frequency Spectrum", color="#cccccc", fontsize=9)
            axes[1, 1].set_xlabel("Freq (Hz)", color="#888888", fontsize=7)
            axes[1, 1].legend(fontsize=7, facecolor="#1a1a1a", edgecolor="#333333", labelcolor="#cccccc")

            plt.tight_layout(pad=1.5)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", facecolor="#1a1a1a", bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            pil_image = Image.open(buf)

            ctk_image = ctk.CTkImage(light_image=pil_image, size=pil_image.size)
            img_label = ctk.CTkLabel(self.results_frame, image=ctk_image, text="")
            img_label.pack(padx=10, pady=(5, 10))
            img_label._ctk_image = ctk_image

            self.log.success("Visualizations generated")

        except Exception as e:
            self.log.error(f"Visualization error: {e}")

    def _show_noise_reduction(self, before, after, sr):
        """Show before/after noise reduction comparison."""
        self.log.info("Generating noise reduction visualization...")

        # Destroy previous NR frame if exists
        if hasattr(self, 'nr_frame') and self.nr_frame is not None:
            self.nr_frame.destroy()

        self.nr_frame = ctk.CTkFrame(self.main)
        # Insert after selector
        self.nr_frame.pack(pady=5, padx=15, fill="x", after=self.selector_frame)

        ctk.CTkLabel(
            self.nr_frame, text="Noise Reduction Analysis",
            font=("Arial", 14, "bold")
        ).pack(pady=(8, 5))

        try:
            # Calculate noise reduction metrics
            before_rms = float(np.sqrt(np.mean(before ** 2)))
            after_rms = float(np.sqrt(np.mean(after ** 2)))
            noise_removed = before - after[:len(before)]
            noise_rms = float(np.sqrt(np.mean(noise_removed ** 2)))

            noise_reduction_db = 0
            if noise_rms > 0 and before_rms > 0:
                noise_reduction_db = 20 * np.log10(before_rms / max(noise_rms, 1e-10))

            # Gauges
            nr_gauges = ctk.CTkFrame(self.nr_frame)
            nr_gauges.pack(pady=3, padx=10, fill="x")
            nr_gauges.grid_columnconfigure(0, weight=1)
            nr_gauges.grid_columnconfigure(1, weight=1)
            nr_gauges.grid_columnconfigure(2, weight=1)

            # Noise removed
            nr_pct = min(1.0, noise_rms / max(before_rms, 0.001))
            nr_label = f"{nr_pct:.0%} removed"
            nr_color = "#2ecc71" if nr_pct > 0.3 else "#f39c12" if nr_pct > 0.1 else "#3498db"
            self._build_gauge(nr_gauges, 0, "Noise Removed", nr_pct, nr_label, nr_color)

            # SNR improvement
            snr_before = 20 * np.log10(before_rms / max(noise_rms, 1e-10)) if noise_rms > 0 else 40
            snr_pct = min(1.0, max(0, snr_before / 40))
            snr_label = f"+{noise_reduction_db:.1f} dB"
            snr_color = "#2ecc71" if noise_reduction_db > 5 else "#f39c12" if noise_reduction_db > 2 else "#888888"
            self._build_gauge(nr_gauges, 1, "SNR Improvement", snr_pct, snr_label, snr_color)

            # Signal preserved
            if before_rms > 0:
                preserved = after_rms / before_rms
            else:
                preserved = 1.0
            pres_pct = min(1.0, preserved)
            pres_label = f"{preserved:.0%} kept"
            pres_color = "#2ecc71" if preserved > 0.7 else "#f39c12" if preserved > 0.4 else "#e74c3c"
            self._build_gauge(nr_gauges, 2, "Signal Preserved", pres_pct, pres_label, pres_color)

            self.log.success(f"Noise reduction: {nr_pct:.0%} noise removed, {snr_label} improvement, {preserved:.0%} signal preserved")

            # Plots: before/after waveform + spectrograms + noise signal
            fig, axes = plt.subplots(2, 2, figsize=(10, 4), dpi=85)
            fig.patch.set_facecolor("#1a1a1a")

            for ax in axes.flat:
                ax.set_facecolor("#1a1a1a")
                ax.tick_params(colors="#888888", labelsize=7)
                for spine in ax.spines.values():
                    spine.set_color("#333333")

            dur = len(before) / sr
            t = np.linspace(0, dur, len(before))

            # Before waveform
            axes[0, 0].plot(t, before, color="#e74c3c", linewidth=0.5)
            axes[0, 0].set_title("Before Noise Reduction", color="#e74c3c", fontsize=9)
            axes[0, 0].set_ylabel("Amplitude", color="#888888", fontsize=7)

            # After waveform
            t_after = np.linspace(0, len(after)/sr, len(after))
            axes[0, 1].plot(t_after, after, color="#2ecc71", linewidth=0.5)
            axes[0, 1].set_title("After Noise Reduction", color="#2ecc71", fontsize=9)
            axes[0, 1].set_ylabel("Amplitude", color="#888888", fontsize=7)

            # Before spectrogram
            spec_before = librosa.feature.melspectrogram(y=before, sr=sr, n_mels=80)
            db_before = librosa.power_to_db(spec_before, ref=np.max)
            axes[1, 0].imshow(db_before, aspect="auto", origin="lower", cmap="inferno",
                             extent=[0, dur, 0, sr/2000])
            axes[1, 0].set_title("Before (Spectrogram)", color="#e74c3c", fontsize=9)
            axes[1, 0].set_xlabel("Time (s)", color="#888888", fontsize=7)
            axes[1, 0].set_ylabel("Freq (kHz)", color="#888888", fontsize=7)

            # After spectrogram
            spec_after = librosa.feature.melspectrogram(y=after, sr=sr, n_mels=80)
            db_after = librosa.power_to_db(spec_after, ref=np.max)
            axes[1, 1].imshow(db_after, aspect="auto", origin="lower", cmap="magma",
                             extent=[0, len(after)/sr, 0, sr/2000])
            axes[1, 1].set_title("After (Spectrogram)", color="#2ecc71", fontsize=9)
            axes[1, 1].set_xlabel("Time (s)", color="#888888", fontsize=7)
            axes[1, 1].set_ylabel("Freq (kHz)", color="#888888", fontsize=7)

            plt.tight_layout(pad=1.5)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", facecolor="#1a1a1a", bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            pil_image = Image.open(buf)

            ctk_image = ctk.CTkImage(light_image=pil_image, size=pil_image.size)
            img_label = ctk.CTkLabel(self.nr_frame, image=ctk_image, text="")
            img_label.pack(padx=10, pady=(5, 10))
            img_label._ctk_image = ctk_image

            self.log.success("Noise reduction visualization generated")

        except Exception as e:
            self.log.error(f"NR visualization error: {e}")

    def _build_gauge(self, parent, col, title, value, label, color):
        """Build a visual quality gauge with progress bar."""
        frame = ctk.CTkFrame(parent)
        frame.grid(row=0, column=col, padx=5, pady=5, sticky="nsew")

        ctk.CTkLabel(
            frame, text=title, font=("Arial", 11), text_color="gray"
        ).pack(pady=(8, 2))

        ctk.CTkLabel(
            frame, text=label, font=("Arial", 18, "bold"), text_color=color
        ).pack(pady=(0, 3))

        bar = ctk.CTkProgressBar(frame, width=120, height=10)
        bar.pack(pady=(0, 3))
        bar.set(max(0.01, min(1.0, value)))
        bar.configure(progress_color=color)

        ctk.CTkLabel(
            frame, text=f"{value:.0%}", font=("Arial", 10), text_color="gray"
        ).pack(pady=(0, 8))

    # ── Audio Selector ──

    def _show_selector(self):
        """Build a simple audio position selector with preview."""
        if self.selector_visible:
            self.selector_frame.destroy()

        self.selector_frame = ctk.CTkFrame(self.main)
        self.selector_frame.pack(pady=5, padx=15, fill="x",
                                  after=self.source_tabs)
        self.selector_visible = True

        ctk.CTkLabel(
            self.selector_frame,
            text="Select a speech region to clone",
            font=("Arial", 13, "bold")
        ).pack(pady=(8, 3))

        # Waveform canvas
        self.waveform_canvas = Canvas(
            self.selector_frame, height=100, bg="#1a1a1a",
            highlightthickness=0, cursor="hand2"
        )
        self.waveform_canvas.pack(padx=10, pady=5, fill="x")
        self.waveform_canvas.bind("<Configure>", lambda e: self._draw_waveform())
        self.waveform_canvas.bind("<ButtonPress-1>", self._on_canvas_click)

        # Duration slider
        dur_row = ctk.CTkFrame(self.selector_frame, fg_color="transparent")
        dur_row.pack(padx=10, pady=(5, 2), fill="x")

        ctk.CTkLabel(dur_row, text="Length:", width=60).pack(side="left")
        self.sel_duration = 5.0
        self.dur_slider = ctk.CTkSlider(
            dur_row, from_=1, to=min(self.raw_duration, 30),
            command=self._on_dur_change
        )
        self.dur_slider.pack(side="left", expand=True, fill="x", padx=5)
        self.dur_slider.set(5.0)
        self.dur_label = ctk.CTkLabel(dur_row, text="5.0s", width=45, text_color="cyan")
        self.dur_label.pack(side="left")

        # Position slider
        pos_row = ctk.CTkFrame(self.selector_frame, fg_color="transparent")
        pos_row.pack(padx=10, pady=2, fill="x")

        ctk.CTkLabel(pos_row, text="Position:", width=60).pack(side="left")
        self.pos_slider = ctk.CTkSlider(
            pos_row, from_=0, to=max(self.raw_duration - self.sel_duration, 0.1),
            command=self._on_pos_change
        )
        self.pos_slider.pack(side="left", expand=True, fill="x", padx=5)
        self.pos_slider.set(self.sel_start)
        self.pos_label = ctk.CTkLabel(
            pos_row,
            text=f"{self.sel_start:.1f}s — {self.sel_end:.1f}s",
            width=120, text_color="cyan"
        )
        self.pos_label.pack(side="left")

        # Buttons
        btn_row = ctk.CTkFrame(self.selector_frame, fg_color="transparent")
        btn_row.pack(padx=10, pady=(3, 8), fill="x")

        ctk.CTkButton(
            btn_row, text="Preview", width=100,
            command=self._on_preview, fg_color="#3498db", hover_color="#2980b9"
        ).pack(side="left", padx=(0, 5))

        ctk.CTkButton(
            btn_row, text="Stop", width=70,
            command=lambda: sd.stop(), fg_color="#e74c3c", hover_color="#c0392b"
        ).pack(side="left", padx=(0, 5))

        self.use_btn = ctk.CTkButton(
            btn_row, text="Use This Region", width=140,
            command=self._on_use_selection, fg_color="#2ecc71", hover_color="#27ae60"
        )
        self.use_btn.pack(side="left", padx=(0, 5))

        self.sel_status = ctk.CTkLabel(
            btn_row, text="", text_color="gray", font=("Arial", 11)
        )
        self.sel_status.pack(side="left", padx=10)

    def _on_canvas_click(self, event):
        """Click on waveform to set position and preview from there."""
        w = self.waveform_canvas.winfo_width()
        if w <= 0 or self.raw_duration <= 0:
            return
        click_time = max(0, min(self.raw_duration - self.sel_duration, (event.x / w) * self.raw_duration))
        self.sel_start = click_time
        self.sel_end = min(click_time + self.sel_duration, self.raw_duration)
        self.pos_slider.set(self.sel_start)
        self._update_pos_label()
        self._draw_waveform()
        self._on_preview()

    def _on_dur_change(self, value):
        self.sel_duration = float(value)
        self.dur_label.configure(text=f"{self.sel_duration:.1f}s")
        # Update position slider max
        self.pos_slider.configure(to=max(self.raw_duration - self.sel_duration, 0.1))
        # Recalculate end
        self.sel_end = min(self.sel_start + self.sel_duration, self.raw_duration)
        self._update_pos_label()
        self._draw_waveform()

    def _on_pos_change(self, value):
        self.sel_start = float(value)
        self.sel_end = min(self.sel_start + self.sel_duration, self.raw_duration)
        self._update_pos_label()
        self._draw_waveform()

    def _update_pos_label(self):
        self.pos_label.configure(text=f"{self.sel_start:.1f}s — {self.sel_end:.1f}s")

    def _on_preview(self):
        """Play the current 5-second selection."""
        if self.raw_audio is None:
            return
        sd.stop()
        start = int(self.sel_start * self.raw_sr)
        end = int(self.sel_end * self.raw_sr)
        region = self.raw_audio[start:end]
        if len(region) > 0:
            self.sel_status.configure(text="Playing...", text_color="cyan")
            def play():
                sd.play(region, self.raw_sr)
                sd.wait()
                self.after(0, lambda: self.sel_status.configure(text=""))
            threading.Thread(target=play, daemon=True).start()

    def _draw_waveform(self):
        canvas = self.waveform_canvas
        canvas.delete("all")
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        if w < 10 or self.raw_audio is None:
            return

        audio = self.raw_audio
        duration = self.raw_duration
        num_bars = min(w, 500)
        samples_per_bar = max(1, len(audio) // num_bars)
        bar_width = w / num_bars
        mid_y = h // 2

        # Selection background
        sx1 = (self.sel_start / duration) * w
        sx2 = (self.sel_end / duration) * w
        canvas.create_rectangle(sx1, 0, sx2, h, fill="#1a3a1a", outline="")

        for i in range(num_bars):
            idx = i * samples_per_bar
            chunk = audio[idx:idx + samples_per_bar]
            if len(chunk) == 0:
                continue
            amp = float(np.max(np.abs(chunk)))
            bar_h = max(1, int(amp * mid_y * 0.85))
            t = (i / num_bars) * duration
            x = i * bar_width
            color = "#2ecc71" if self.sel_start <= t <= self.sel_end else "#444444"
            canvas.create_line(x, mid_y - bar_h, x, mid_y + bar_h,
                             fill=color, width=max(1, bar_width * 0.8))

        # Boundary lines
        canvas.create_line(sx1, 0, sx1, h, fill="#e74c3c", width=2)
        canvas.create_line(sx2, 0, sx2, h, fill="#e74c3c", width=2)

        # Time labels on boundaries
        canvas.create_text(sx1 + 3, 10, text=f"{self.sel_start:.1f}s",
                          fill="#e74c3c", anchor="w", font=("Arial", 8))
        canvas.create_text(sx2 - 3, 10, text=f"{self.sel_end:.1f}s",
                          fill="#e74c3c", anchor="e", font=("Arial", 8))

    def _on_use_selection(self):
        """Clean the selected region and set it as reference."""
        if self.raw_audio is None:
            return

        start = int(self.sel_start * self.raw_sr)
        end = int(self.sel_end * self.raw_sr)
        selection = self.raw_audio[start:end]

        self.sel_status.configure(text="Processing...", text_color="yellow")
        self.use_btn.configure(state="disabled")

        def process():
            try:
                from app.utils.file_utils import generate_output_path
                prepared_path = generate_output_path("_ref_prepared.wav")

                self.log.info("Trimming silence...")
                trimmed, _ = librosa.effects.trim(selection, top_db=20)
                before_nr = trimmed.copy()

                self.log.info("Reducing background noise...")
                cleaned = nr.reduce_noise(y=trimmed, sr=self.raw_sr, prop_decrease=0.8)

                # Show noise reduction visualization
                self.after(0, lambda: self._show_noise_reduction(
                    before_nr, cleaned, self.raw_sr
                ))

                # Transcribe the FULL cleaned audio first (more context = better accuracy)
                self.after(0, lambda: self.sel_status.configure(
                    text="Transcribing full clip...", text_color="yellow"
                ))
                self.log.info(f"Transcribing full clip ({len(cleaned)/self.raw_sr:.1f}s) for better accuracy...")

                full_path = generate_output_path("_ref_full.wav")
                sf.write(full_path, cleaned, self.raw_sr)

                import mlx_whisper
                result = mlx_whisper.transcribe(full_path)
                ref_text = result.get("text", "").strip()
                self.log.success(f"Full transcription: \"{ref_text}\"")

                # f5-tts works best with 6-8s reference
                max_ref_samples = int(8.0 * self.raw_sr)
                if len(cleaned) > max_ref_samples:
                    self.log.info(f"Trimming audio to 8s for cloning model. Full: {len(cleaned)/self.raw_sr:.1f}s")
                    best_start = 0
                    best_rms = 0
                    hop = self.raw_sr
                    for s in range(0, len(cleaned) - max_ref_samples, hop):
                        rms = float(np.sqrt(np.mean(cleaned[s:s+max_ref_samples]**2)))
                        if rms > best_rms:
                            best_rms = rms
                            best_start = s
                    cleaned = cleaned[best_start:best_start + max_ref_samples]

                sf.write(prepared_path, cleaned, self.raw_sr)
                dur = len(cleaned) / self.raw_sr
                self.log.success(f"Reference audio: {dur:.1f}s")

                self.voice_module.reference_path = prepared_path
                self.voice_module.reference_text = ref_text
                self.use_reference = True

                self.after(0, lambda: self.sel_status.configure(
                    text=f"Voice captured! Now type text and hit Generate & Play",
                    text_color="#2ecc71"
                ))
                self.after(0, lambda: self.profile_label.configure(
                    text=f"Voice loaded ({dur:.1f}s) | Heard: \"{ref_text[:80]}\"",
                    text_color="cyan"
                ))
            except Exception as e:
                self.log.error(f"Reference processing failed: {e}")
                self.after(0, lambda: self.sel_status.configure(
                    text=f"Error: {e}", text_color="red"
                ))
            finally:
                self.after(0, lambda: self.use_btn.configure(state="normal"))

        threading.Thread(target=process, daemon=True).start()

    # ── Upload ──

    def _on_upload_voice_sample(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Audio Files", "*.mp3 *.wav")]
        )
        if not file_path:
            return

        self.profile_label.configure(text="Loading audio...", text_color="yellow")
        self.upload_button.configure(state="disabled")

        def load():
            try:
                self.log.info(f"Loading audio: {os.path.basename(file_path)}")
                audio, sr = librosa.load(file_path, sr=24000, mono=True)
                duration = len(audio) / sr
                filename = os.path.basename(file_path)

                self.raw_audio = audio
                self.raw_sr = sr
                self.raw_duration = duration
                self.raw_path = file_path

                self.sel_start = 0.0
                self.sel_end = min(5.0, duration)

                self.log.success(f"Audio loaded: {filename} ({duration:.1f}s, {sr}Hz)")

                self.after(0, lambda: self.profile_label.configure(
                    text=f"{filename} ({duration:.1f}s) — Select a region below",
                    text_color="cyan"
                ))
                self.after(0, self._show_selector)
                self.after(0, lambda: self.status.configure(
                    text="Audio loaded! Select a clean speech region and click 'Use Selection'.",
                    text_color="green"
                ))
            except Exception as e:
                self.after(0, lambda: self.profile_label.configure(
                    text=f"Error: {e}", text_color="red"
                ))
            finally:
                self.after(0, lambda: self.upload_button.configure(state="normal"))

        threading.Thread(target=load, daemon=True).start()

    # ── Generate ──

    def _on_generate(self):
        text = self.text_input.get("1.0", "end").strip()
        if not text:
            self.status.configure(text="Please enter some text.", text_color="red")
            return

        current_tab = self.source_tabs.get()
        is_upload = current_tab == "Upload Sample"

        if is_upload and not self.use_reference:
            self.status.configure(text="Upload audio and click 'Use Selection' first.", text_color="red")
            return

        msg = "Cloning voice (~15s)..." if is_upload else "Generating..."
        self._start_loading(msg)

        pitch = int(self.pitch_slider.get())
        speed = round(self.speed_slider.get(), 2)
        preset = self.preset_selector.get()

        ref_path = None
        if is_upload and self.voice_module.reference_path:
            ref_path = self.voice_module.reference_path

        def run():
            try:
                if is_upload:
                    self.log.info(f"Cloning voice: \"{text[:50]}...\"")
                    output_path = self.voice_module.synthesize_from_reference(text)
                    self.log.success(f"Voice cloned: {os.path.basename(output_path)}")
                else:
                    self.log.info(f"Generating preset '{preset}': \"{text[:50]}...\"")
                    output_path = self.voice_module.synthesize_from_preset(text, preset)
                    self.log.success(f"Preset generated: {os.path.basename(output_path)}")

                if pitch != 0 or speed != 1.0:
                    self.log.info(f"Applying adjustments: pitch={pitch}, speed={speed}x")
                    output_path = self._apply_adjustments(output_path, pitch, speed)

                self.after(0, lambda: self._stop_loading())
                self.after(0, lambda: self.status.configure(
                    text="Playing...", text_color="cyan"
                ))
                self._play_audio(output_path)

                self.after(0, lambda: self._show_replay(output_path))
                self.after(0, lambda: self.status.configure(
                    text=f"Done! Saved: {os.path.basename(output_path)}",
                    text_color="green"
                ))

                viz_ref = ref_path if ref_path else output_path
                self.after(0, lambda: self._show_results(viz_ref, output_path))

            except Exception as e:
                self.log.error(f"Generation failed: {e}")
                self.after(0, lambda: self._stop_loading())
                self.after(0, lambda: self.status.configure(
                    text=f"Error: {e}", text_color="red"
                ))

        threading.Thread(target=run, daemon=True).start()

    def _apply_adjustments(self, audio_path, pitch_semitones, speed_factor):
        from app.utils.file_utils import generate_output_path

        audio, sr = librosa.load(audio_path, sr=DEFAULT_SAMPLE_RATE)
        if pitch_semitones != 0:
            audio = librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=pitch_semitones)
        if speed_factor != 1.0:
            audio = librosa.effects.time_stretch(y=audio, rate=speed_factor)
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.9
        out = generate_output_path("adjusted_voice.wav")
        sf.write(out, audio, sr)
        return out

    def _play_audio(self, file_path):
        data, sr = sf.read(file_path)
        sd.play(data, sr)
        sd.wait()
