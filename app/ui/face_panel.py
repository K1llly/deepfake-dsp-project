import os
import io
import threading
import time
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.modules.input_module import InputModule, InputValidationError
from app.modules.face_swap import FaceSwapModule
from app.modules.output_module import OutputModule
from app.modules.evaluation import EvaluationModule
from app.utils.logger import get_logger
import mediapipe as mp
from config import MODELS_DIR


class FacePanel(ctk.CTkFrame):
    """UI panel for live webcam face swapping."""

    def __init__(self, parent):
        super().__init__(parent)

        self.input_module = InputModule()
        self.face_swap_module = FaceSwapModule()
        self.output_module = OutputModule()
        self.evaluation_module = EvaluationModule()
        self.log = get_logger()

        # Face detector for bounding box + confidence
        detector_model = os.path.join(MODELS_DIR, "face_detector.tflite")
        detector_options = mp.tasks.vision.FaceDetectorOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=detector_model),
            min_detection_confidence=0.3,
        )
        self.face_detector = mp.tasks.vision.FaceDetector.create_from_options(
            detector_options
        )

        self.source_image = None
        self.webcam_running = False
        self.capture = None
        self.last_swapped_frame = None
        self.last_original_frame = None
        self.fps = 0

        self._build_layout()

    def _build_layout(self):
        """Construct the live face swap panel."""
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Scrollable container
        self.scroll = ctk.CTkScrollableFrame(self)
        self.scroll.grid(row=0, column=0, sticky="nsew")
        self.scroll.grid_columnconfigure(0, weight=1)

        # --- Top Controls ---
        controls = ctk.CTkFrame(self.scroll)
        controls.pack(padx=10, pady=(10, 5), fill="x")
        controls.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            controls, text="Live Face Swap", font=("Arial", 20, "bold")
        ).grid(row=0, column=0, columnspan=4, pady=(10, 8))

        self.load_face_button = ctk.CTkButton(
            controls, text="Upload Face Image",
            command=self._on_load_source
        )
        self.load_face_button.grid(row=1, column=0, padx=10, pady=8)

        self.source_name_label = ctk.CTkLabel(
            controls, text="No face loaded", text_color="gray"
        )
        self.source_name_label.grid(row=1, column=1, padx=5, sticky="w")

        self.start_button = ctk.CTkButton(
            controls, text="Start Webcam",
            command=self._on_start_webcam,
            fg_color="#2ecc71", hover_color="#27ae60"
        )
        self.start_button.grid(row=1, column=2, padx=5, pady=8)

        self.stop_button = ctk.CTkButton(
            controls, text="Stop Webcam",
            command=self._on_stop_webcam,
            fg_color="#e74c3c", hover_color="#c0392b",
            state="disabled"
        )
        self.stop_button.grid(row=1, column=3, padx=(5, 10), pady=8)

        # --- Live Video Panels ---
        video_container = ctk.CTkFrame(self.scroll)
        video_container.pack(padx=10, pady=5, fill="x")
        video_container.grid_columnconfigure(0, weight=1)
        video_container.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            video_container, text="Webcam (Face Detection)",
            font=("Arial", 14, "bold")
        ).grid(row=0, column=0, pady=(10, 2))

        ctk.CTkLabel(
            video_container, text="Live Face Swap",
            font=("Arial", 14, "bold")
        ).grid(row=0, column=1, pady=(10, 2))

        self.webcam_label = ctk.CTkLabel(
            video_container, text="Webcam Off", fg_color="#1a1a1a",
            width=480, height=320
        )
        self.webcam_label.grid(row=1, column=0, padx=8, pady=8)

        self.swapped_label = ctk.CTkLabel(
            video_container, text="Waiting...", fg_color="#1a1a1a",
            width=480, height=320
        )
        self.swapped_label.grid(row=1, column=1, padx=8, pady=8)

        # --- Bottom Bar ---
        bottom = ctk.CTkFrame(self.scroll)
        bottom.pack(padx=10, pady=(5, 10), fill="x")
        bottom.grid_columnconfigure(1, weight=1)

        self.capture_button = ctk.CTkButton(
            bottom, text="Capture Screenshot",
            command=self._on_capture_screenshot, state="disabled"
        )
        self.capture_button.grid(row=0, column=0, padx=10, pady=8)

        self.fps_label = ctk.CTkLabel(
            bottom, text="FPS: --", text_color="gray"
        )
        self.fps_label.grid(row=0, column=1, padx=5, sticky="w")

        self.psnr_label = ctk.CTkLabel(
            bottom, text="PSNR: --", text_color="cyan", font=("Arial", 12)
        )
        self.psnr_label.grid(row=0, column=2, padx=5)

        self.ssim_label = ctk.CTkLabel(
            bottom, text="SSIM: --", text_color="cyan", font=("Arial", 12)
        )
        self.ssim_label.grid(row=0, column=3, padx=5)

        self.status_label = ctk.CTkLabel(bottom, text="", text_color="gray")
        self.status_label.grid(row=0, column=4, padx=10, sticky="e")

    # ── Controls ──

    def _on_load_source(self):
        """Load the face image to overlay."""
        file_path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png")]
        )
        if not file_path:
            return

        try:
            self.source_image = self.input_module.load_image(file_path)
            filename = os.path.basename(file_path)
            self.source_name_label.configure(
                text=f"Loading face...", text_color="yellow"
            )
            self.load_face_button.configure(state="disabled")
            self.log.info(f"Detecting face in: {filename}")

            def detect():
                try:
                    found = self.face_swap_module.set_source_face(self.source_image)
                    if found:
                        self.after(0, lambda: self.source_name_label.configure(
                            text=f"Loaded: {filename}", text_color="cyan"
                        ))
                        self.after(0, lambda: self.status_label.configure(
                            text="Face detected! Start webcam to begin.", text_color="green"
                        ))
                        self.log.success(f"Source face detected and cached: {filename}")
                    else:
                        self.after(0, lambda: self.source_name_label.configure(
                            text="No face found in image", text_color="red"
                        ))
                        self.log.error(f"No face detected in: {filename}")
                        self.source_image = None
                except Exception as e:
                    self.log.error(f"Face detection error: {e}")
                    self.after(0, lambda: self.source_name_label.configure(
                        text=f"Error: {e}", text_color="red"
                    ))
                finally:
                    self.after(0, lambda: self.load_face_button.configure(state="normal"))

            threading.Thread(target=detect, daemon=True).start()

        except InputValidationError as error:
            self.status_label.configure(text=str(error), text_color="red")
            self.load_face_button.configure(state="normal")
            self.log.error(f"Face load failed: {error}")

    def _on_start_webcam(self):
        """Start the webcam feed."""
        if self.source_image is None:
            self.status_label.configure(
                text="Upload a face image first!", text_color="red"
            )
            return

        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            self.status_label.configure(
                text="Could not open webcam.", text_color="red"
            )
            return

        self.webcam_running = True
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.capture_button.configure(state="normal")
        self.status_label.configure(text="Webcam running...", text_color="green")
        self.log.success("Webcam started")

        threading.Thread(target=self._webcam_loop, daemon=True).start()

    def _on_stop_webcam(self):
        """Stop the webcam feed."""
        self.webcam_running = False
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.capture_button.configure(state="disabled")
        self.status_label.configure(text="Webcam stopped.", text_color="gray")
        self.fps_label.configure(text="FPS: --")

        if self.capture is not None:
            self.capture.release()
            self.capture = None

        self.webcam_label.configure(image=None, text="Webcam Off")
        self.swapped_label.configure(image=None, text="Waiting...")

    def _on_capture_screenshot(self):
        """Save the current swapped frame and show analysis."""
        if self.last_swapped_frame is None:
            return

        # Grab current original frame too
        if self.capture is not None and self.capture.isOpened():
            ret, current_frame = self.capture.read()
            if ret:
                current_frame = cv2.flip(current_frame, 1)
                self.last_original_frame = current_frame

        output_path = self.output_module.save_image(
            self.last_swapped_frame, "live_swap_capture.png"
        )
        self.status_label.configure(
            text=f"Saved: {os.path.basename(output_path)}", text_color="green"
        )
        self.log.success(f"Screenshot saved: {os.path.basename(output_path)}")

        # Show analysis
        self._show_face_analysis()

    def _show_face_analysis(self):
        """Show face detection visualization and quality metrics."""
        if self.last_original_frame is None or self.source_image is None:
            return

        self.log.info("Generating face swap analysis...")

        # Run a fresh swap on the captured frame for consistent analysis
        original = self.last_original_frame.copy()
        swapped = self.face_swap_module.swap_faces(self.source_image, original)
        if swapped is None:
            swapped = self.last_swapped_frame if self.last_swapped_frame is not None else original
            self.log.warning("Could not swap for analysis, using last cached result")

        self.last_swapped_frame = swapped

        # Create or replace analysis frame
        if hasattr(self, 'analysis_frame') and self.analysis_frame is not None:
            self.analysis_frame.destroy()

        self.analysis_frame = ctk.CTkFrame(self.scroll)
        self.analysis_frame.pack(padx=10, pady=(0, 10), fill="x")

        ctk.CTkLabel(
            self.analysis_frame, text="Face Swap Analysis",
            font=("Arial", 14, "bold")
        ).pack(pady=(8, 5))

        try:
            source = self.source_image

            # Compute metrics
            small_orig = cv2.resize(original, (256, 256))
            small_swap = cv2.resize(swapped, (256, 256))

            psnr = self.evaluation_module.calculate_psnr(small_orig, small_swap)
            ssim = self.evaluation_module.calculate_ssim(small_orig, small_swap)

            # Face detection confidence
            faces = self.face_swap_module.app.get(original)
            detection_conf = faces[0].det_score if faces else 0

            # Gauges
            gauges = ctk.CTkFrame(self.analysis_frame)
            gauges.pack(pady=5, padx=10, fill="x")
            for i in range(4):
                gauges.grid_columnconfigure(i, weight=1)

            # PSNR gauge
            psnr_pct = min(1.0, max(0, psnr / 50)) if psnr != float("inf") else 1.0
            psnr_label = f"{psnr:.1f} dB" if psnr != float("inf") else "Perfect"
            psnr_color = "#2ecc71" if psnr > 30 else "#f39c12" if psnr > 20 else "#e74c3c"
            self._build_gauge(gauges, 0, "PSNR", psnr_pct, psnr_label, psnr_color)

            # SSIM gauge
            ssim_pct = max(0, ssim)
            ssim_label = f"{ssim:.3f}"
            ssim_color = "#2ecc71" if ssim > 0.8 else "#f39c12" if ssim > 0.5 else "#e74c3c"
            self._build_gauge(gauges, 1, "SSIM", ssim_pct, ssim_label, ssim_color)

            # Detection confidence
            conf_pct = float(detection_conf)
            conf_label = f"{conf_pct:.0%}"
            conf_color = "#2ecc71" if conf_pct > 0.8 else "#f39c12" if conf_pct > 0.5 else "#e74c3c"
            self._build_gauge(gauges, 2, "Face Detection", conf_pct, conf_label, conf_color)

            # Swap quality (inverse of difference)
            diff = cv2.absdiff(small_orig, small_swap)
            change_pct = float(np.mean(diff)) / 255
            quality_pct = 1.0 - change_pct
            quality_label = "Seamless" if quality_pct > 0.85 else "Good" if quality_pct > 0.7 else "Visible"
            quality_color = "#2ecc71" if quality_pct > 0.85 else "#f39c12" if quality_pct > 0.7 else "#e74c3c"
            self._build_gauge(gauges, 3, "Blend Quality", quality_pct, quality_label, quality_color)

            self.log.success(f"PSNR: {psnr_label} | SSIM: {ssim_label} | Detection: {conf_label} | Blend: {quality_label}")

            # Visualization plots
            fig, axes = plt.subplots(2, 3, figsize=(11, 5), dpi=85)
            fig.patch.set_facecolor("#1a1a1a")

            for ax in axes.flat:
                ax.set_facecolor("#1a1a1a")
                ax.tick_params(colors="#888888", labelsize=6)
                for spine in ax.spines.values():
                    spine.set_color("#333333")
                ax.set_xticks([])
                ax.set_yticks([])

            # 1. Source face
            src_rgb = cv2.cvtColor(cv2.resize(source, (256, 256)), cv2.COLOR_BGR2RGB)
            axes[0, 0].imshow(src_rgb)
            axes[0, 0].set_title("Source Face", color="#3498db", fontsize=9)

            # 2. Original frame (with face landmarks)
            orig_vis = original.copy()
            if faces:
                kps = faces[0].kps.astype(int)
                for kp in kps:
                    cv2.circle(orig_vis, tuple(kp), 3, (0, 255, 0), -1)
                bbox = faces[0].bbox.astype(int)
                cv2.rectangle(orig_vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            orig_rgb = cv2.cvtColor(cv2.resize(orig_vis, (256, 256)), cv2.COLOR_BGR2RGB)
            axes[0, 1].imshow(orig_rgb)
            axes[0, 1].set_title("Detected Face + Landmarks", color="#2ecc71", fontsize=9)

            # 3. Swapped result
            swap_rgb = cv2.cvtColor(cv2.resize(swapped, (256, 256)), cv2.COLOR_BGR2RGB)
            axes[0, 2].imshow(swap_rgb)
            axes[0, 2].set_title("Swapped Result", color="#e74c3c", fontsize=9)

            # 4. Difference heatmap
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            axes[1, 0].imshow(diff_gray, cmap="hot")
            axes[1, 0].set_title("Difference Heatmap", color="#f39c12", fontsize=9)

            # 5. Face region extraction + 6. Swapped face region
            if faces:
                bbox = faces[0].bbox.astype(int)
                fx1 = max(0, bbox[0])
                fy1 = max(0, bbox[1])
                fx2 = min(original.shape[1], bbox[2])
                fy2 = min(original.shape[0], bbox[3])

                if fx2 > fx1 and fy2 > fy1:
                    face_crop = original[fy1:fy2, fx1:fx2]
                    if face_crop.size > 0:
                        face_rgb = cv2.cvtColor(cv2.resize(face_crop, (256, 256)), cv2.COLOR_BGR2RGB)
                        axes[1, 1].imshow(face_rgb)

                    # Clamp to swapped frame bounds too
                    sx2 = min(swapped.shape[1], fx2)
                    sy2 = min(swapped.shape[0], fy2)
                    if sx2 > fx1 and sy2 > fy1:
                        swap_crop = swapped[fy1:sy2, fx1:sx2]
                        if swap_crop.size > 0:
                            swap_face_rgb = cv2.cvtColor(cv2.resize(swap_crop, (256, 256)), cv2.COLOR_BGR2RGB)
                            axes[1, 2].imshow(swap_face_rgb)

            axes[1, 1].set_title("Extracted Face Region", color="#9b59b6", fontsize=9)
            axes[1, 2].set_title("Swapped Face Region", color="#e74c3c", fontsize=9)

            plt.tight_layout(pad=1.5)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", facecolor="#1a1a1a", bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            pil_image = Image.open(buf)

            ctk_image = ctk.CTkImage(light_image=pil_image, size=pil_image.size)
            img_label = ctk.CTkLabel(self.analysis_frame, image=ctk_image, text="")
            img_label.pack(padx=10, pady=(5, 10))
            img_label._ctk_image = ctk_image

            self.log.success("Face swap analysis generated")

        except Exception as e:
            self.log.error(f"Face analysis error: {e}")

    def _build_gauge(self, parent, col, title, value, label, color):
        """Build a visual quality gauge."""
        frame = ctk.CTkFrame(parent)
        frame.grid(row=0, column=col, padx=5, pady=5, sticky="nsew")

        ctk.CTkLabel(frame, text=title, font=("Arial", 11), text_color="gray").pack(pady=(8, 2))
        ctk.CTkLabel(frame, text=label, font=("Arial", 16, "bold"), text_color=color).pack(pady=(0, 3))

        bar = ctk.CTkProgressBar(frame, width=100, height=10)
        bar.pack(pady=(0, 3))
        bar.set(max(0.01, min(1.0, value)))
        bar.configure(progress_color=color)

        ctk.CTkLabel(frame, text=f"{value:.0%}", font=("Arial", 10), text_color="gray").pack(pady=(0, 8))

    # ── Webcam Loop ──

    def _webcam_loop(self):
        """Webcam loop: smooth display + async face swap in background."""
        frame_count = 0
        psnr_val = "--"
        ssim_val = "--"
        last_swap_result = None
        swap_busy = False
        swap_fps_times = []

        def run_swap(frame_to_swap):
            """Run face swap in background — called from main loop."""
            nonlocal last_swap_result, swap_busy, swap_fps_times
            swap_start = time.time()
            try:
                self.face_swap_module.detect_face(frame_to_swap)
                result = self.face_swap_module.swap_with_cached_face(frame_to_swap)
                if result is not None:
                    last_swap_result = result
                    self.last_swapped_frame = result
            except Exception:
                pass
            swap_elapsed = time.time() - swap_start
            swap_fps_times.append(swap_elapsed)
            if len(swap_fps_times) > 10:
                swap_fps_times.pop(0)
            swap_busy = False

        while self.webcam_running and self.capture is not None:
            success, frame = self.capture.read()
            if not success:
                continue

            h, w = frame.shape[:2]
            if w > 960:
                scale = 960 / w
                frame = cv2.resize(frame, (960, int(h * scale)))

            frame = cv2.flip(frame, 1)
            frame_count += 1
            self.last_original_frame = frame.copy()

            # Annotate webcam (lightweight — just face detection box)
            annotated_frame = self._annotate_webcam_frame(frame.copy())

            # Launch swap in background thread if not busy
            if not swap_busy and self.source_image is not None:
                swap_busy = True
                threading.Thread(
                    target=run_swap, args=(frame.copy(),), daemon=True
                ).start()

            # Use last swap result for display
            display_swap = last_swap_result if last_swap_result is not None else frame

            # Calculate swap FPS
            if swap_fps_times:
                swap_fps = len(swap_fps_times) / sum(swap_fps_times)
            else:
                swap_fps = 0

            # PSNR/SSIM occasionally
            if frame_count % 30 == 0 and last_swap_result is not None:
                try:
                    small_orig = cv2.resize(frame, (128, 96))
                    small_swap = cv2.resize(last_swap_result, (128, 96))
                    psnr_val = self.evaluation_module.calculate_psnr(small_orig, small_swap)
                    ssim_val = self.evaluation_module.calculate_ssim(small_orig, small_swap)
                except Exception:
                    pass

            # Update UI — webcam is smooth, swap updates when ready
            self.after(0, self._update_displays,
                       annotated_frame, display_swap, swap_fps,
                       psnr_val, ssim_val)

            # Smooth webcam at ~25 FPS
            time.sleep(0.04)

    def _annotate_webcam_frame(self, frame):
        """Draw face box, landmarks, confidence, and expression info."""
        face = self.face_swap_module.last_target_face
        if face is None:
            return frame

        # Bounding box
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Confidence
        confidence = float(face.det_score) * 100
        self._draw_label(frame, f"{confidence:.0f}%", x2 - 55, y2 - 5, (0, 255, 0))

        # 5-point landmarks (eyes, nose, mouth corners)
        kps = face.kps.astype(int)
        colors = [(255, 200, 0), (255, 200, 0), (0, 200, 255), (255, 100, 100), (255, 100, 100)]
        labels = ["L.Eye", "R.Eye", "Nose", "L.Mouth", "R.Mouth"]
        for i, (kp, color) in enumerate(zip(kps, colors)):
            cv2.circle(frame, tuple(kp), 4, color, -1)
            cv2.circle(frame, tuple(kp), 6, color, 1)

        # Expression analysis from landmarks
        left_eye = kps[0]
        right_eye = kps[1]
        nose = kps[2]
        left_mouth = kps[3]
        right_mouth = kps[4]

        # Head pose estimation (yaw from eye-nose symmetry)
        eye_center_x = (left_eye[0] + right_eye[0]) / 2
        nose_offset = nose[0] - eye_center_x
        eye_dist = np.linalg.norm(right_eye - left_eye)
        yaw_ratio = nose_offset / max(eye_dist, 1) * 100
        if abs(yaw_ratio) < 8:
            head_dir = "Forward"
        elif yaw_ratio > 0:
            head_dir = "Right"
        else:
            head_dir = "Left"

        # Mouth openness (distance between mouth corners vs eye distance)
        mouth_width = np.linalg.norm(right_mouth - left_mouth)
        mouth_ratio = mouth_width / max(eye_dist, 1)
        mouth_state = "Open" if mouth_ratio > 0.55 else "Closed"

        # Eye openness (vertical distance from eye to nose baseline)
        eye_y_avg = (left_eye[1] + right_eye[1]) / 2
        nose_y = nose[1]
        eye_nose_dist = nose_y - eye_y_avg
        face_height = y2 - y1
        eye_ratio = eye_nose_dist / max(face_height, 1)
        eye_state = "Open" if eye_ratio > 0.15 else "Squint"

        # Draw expression info panel
        info_x = x1
        info_y = y1 - 10
        expressions = [
            f"Head: {head_dir}",
            f"Mouth: {mouth_state}",
            f"Eyes: {eye_state}",
        ]

        for i, expr in enumerate(reversed(expressions)):
            ty = info_y - i * 20
            if ty < 10:
                break
            self._draw_label(frame, expr, info_x, ty, (0, 255, 200))

        return frame

    def _draw_label(self, frame, text, x, y, color):
        """Draw a text label with background on frame."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thick = 1
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)
        cv2.rectangle(frame, (x - 2, y - th - 4), (x + tw + 4, y + baseline + 2), (0, 0, 0), -1)
        cv2.putText(frame, text, (x, y), font, scale, color, thick)

    def _process_swap(self, frame):
        """Run face swap using cached face detection (fast)."""
        if self.source_image is None:
            return frame

        try:
            result = self.face_swap_module.swap_with_cached_face(frame)
            if result is not None:
                self.last_swapped_frame = result
                return result
        except Exception:
            pass

        return frame

    def _update_displays(self, webcam_frame, swapped_frame, fps,
                         psnr="--", ssim="--"):
        """Update video panels, FPS, and metrics on the main thread."""
        if not self.webcam_running:
            return

        self._show_frame(webcam_frame, self.webcam_label)
        self._show_frame(swapped_frame, self.swapped_label)
        self.fps_label.configure(text=f"FPS: {fps:.1f}")

        psnr_text = f"PSNR: {psnr} dB" if psnr != "--" else "PSNR: --"
        ssim_text = f"SSIM: {ssim}" if ssim != "--" else "SSIM: --"
        self.psnr_label.configure(text=psnr_text)
        self.ssim_label.configure(text=ssim_text)

    def _show_frame(self, cv2_frame, label_widget):
        """Convert an OpenCV frame and display it in a CTk label."""
        rgb = cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        display_w = max(label_widget.winfo_width(), 320)
        display_h = max(label_widget.winfo_height(), 240)

        width, height = pil_image.size
        scale = min(display_w / width, display_h / height, 1.0)
        new_size = (int(width * scale), int(height * scale))
        pil_image = pil_image.resize(new_size, Image.LANCZOS)

        ctk_image = ctk.CTkImage(light_image=pil_image, size=new_size)
        label_widget.configure(image=ctk_image, text="")
        label_widget._ctk_image = ctk_image

    def destroy(self):
        """Clean up webcam on panel destroy."""
        self.webcam_running = False
        if self.capture is not None:
            self.capture.release()
        super().destroy()
