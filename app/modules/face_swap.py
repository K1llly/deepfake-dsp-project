import os
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.app.common import Face

from config import MODELS_DIR

LIVE_PROCESS_WIDTH = 640


class FaceSwapModule:
    """High-quality face swapping using InsightFace + inswapper + expanded blend."""

    def __init__(self):
        # buffalo_s is ~3-5x faster than buffalo_l with negligible quality
        # loss for swap purposes. Limit to detection + recognition only:
        # FaceAnalysis runs 2D106 / 3D68 landmark + gender/age models per
        # face by default, none of which the swapper consumes — recognition
        # is needed because inswapper reads source_face.normed_embedding.
        # Detection stays on CPU: CoreML mis-infers the RetinaFace output
        # shape and crashes at inference time on our ORT version.
        self.app = FaceAnalysis(
            name="buffalo_s",
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=0, det_size=(160, 160))

        # Swapper on CoreML (Neural Engine) for speed, fallback to CPU
        swapper_path = os.path.join(MODELS_DIR, "inswapper_128.onnx")
        try:
            self.swapper = insightface.model_zoo.get_model(
                swapper_path, providers=["CoreMLExecutionProvider", "CPUExecutionProvider"]
            )
        except Exception:
            self.swapper = insightface.model_zoo.get_model(
                swapper_path, providers=["CPUExecutionProvider"]
            )

        self.source_face = None
        self.last_target_face = None
        # Pre-computed feathered blend mask (3-channel float). Rebuilt on
        # each successful detection so the live preview can blend with one
        # multiply-add per frame instead of redoing GaussianBlur.
        self._cached_blend_mask = None
        # First-order IIR low-pass filter on the detection signal — kills
        # the 1-2 px wobble caused by per-frame detection noise. Reset
        # automatically when the detection jumps farther than half a face
        # height (face left and re-entered, or a different person).
        self._bbox_smoothed = None
        self._kps_smoothed = None

        # Warmup: trigger ORT/CoreML JIT so the first real swap doesn't
        # eat ~400 ms after the user clicks Start.
        self._warmup_swapper()

    def _warmup_swapper(self):
        """Run one throwaway inswapper inference at startup to JIT the graph."""
        try:
            dummy = np.full((256, 256, 3), 128, dtype=np.uint8)
            kps = np.array(
                [[80, 100], [176, 100], [128, 140], [90, 180], [166, 180]],
                dtype=np.float32,
            )
            target = Face(
                bbox=np.array([40, 60, 216, 220], dtype=np.float32),
                kps=kps, det_score=0.9,
            )
            source = Face(
                bbox=np.array([40, 60, 216, 220], dtype=np.float32),
                kps=kps, det_score=0.9,
            )
            emb = np.random.randn(512).astype(np.float32)
            source.normed_embedding = emb / (np.linalg.norm(emb) + 1e-6)
            self.swapper.get(dummy, target, source, paste_back=True)
        except Exception:
            pass

    def set_source_face(self, image):
        """Detect and cache the source face from an image."""
        faces = self.app.get(image)
        if not faces:
            return False
        self.source_face = faces[0]
        return True

    def swap_faces(self, source_image, target_image):
        """Swap source face onto target image (full quality)."""
        if self.source_face is None:
            if not self.set_source_face(source_image):
                return None

        target_faces = self.app.get(target_image)
        if not target_faces:
            return None

        result = target_image.copy()
        for face in target_faces:
            result = self.swapper.get(result, face, self.source_face, paste_back=True)
            result = self._expanded_blend(target_image, result, face)

        return result

    def detect_face(self, frame):
        """Detect face in frame and cache it. Returns True if found.

        Calls the RetinaFace detector directly instead of FaceAnalysis.get(),
        which would also run the recognition model (~30 ms) on every target
        face — but inswapper only uses target.kps, so the embedding is
        wasted work. Source face still uses the full pipeline so its
        embedding is populated for the swapper.

        Applies an EMA low-pass filter to bbox and kps so per-frame
        detection noise doesn't make the swap wobble.
        """
        small = self._downscale(frame)
        det, kpss = self.app.det_model.detect(small, max_num=1)
        if len(det) == 0:
            return False
        scale = frame.shape[1] / small.shape[1]
        new_bbox = det[0, :4] * scale
        score = float(det[0, 4])
        new_kps = kpss[0] * scale if kpss is not None else None

        alpha = 0.4  # weight on the new measurement
        if self._bbox_smoothed is None:
            self._bbox_smoothed = new_bbox.copy()
            self._kps_smoothed = new_kps.copy() if new_kps is not None else None
        else:
            face_h = self._bbox_smoothed[3] - self._bbox_smoothed[1]
            jump = float(np.linalg.norm(new_bbox[:2] - self._bbox_smoothed[:2]))
            if jump > face_h * 0.5:
                # Detection jumped — face re-entered or different person; reset.
                self._bbox_smoothed = new_bbox.copy()
                self._kps_smoothed = new_kps.copy() if new_kps is not None else None
            else:
                self._bbox_smoothed = (
                    (1 - alpha) * self._bbox_smoothed + alpha * new_bbox
                )
                if new_kps is not None and self._kps_smoothed is not None:
                    self._kps_smoothed = (
                        (1 - alpha) * self._kps_smoothed + alpha * new_kps
                    )

        face = Face(
            bbox=self._bbox_smoothed.copy(),
            kps=self._kps_smoothed.copy() if self._kps_smoothed is not None else None,
            det_score=score,
        )
        self.last_target_face = face
        self._cached_blend_mask = self._build_blend_mask(frame.shape, face)
        return True

    def swap_with_cached_face(self, frame, fast=False):
        """Swap using the last cached target face detection.

        fast=True uses the pre-computed feathered mask from the most recent
        detect_face call — same blend quality as the full path, but the
        per-frame cost is one elementwise multiply-add instead of a fresh
        GaussianBlur. Falls through to a raw paste_back result if no mask
        has been cached yet (first frame after start).
        """
        if self.source_face is None or self.last_target_face is None:
            return frame

        original = frame.copy()
        result = self.swapper.get(
            frame, self.last_target_face, self.source_face, paste_back=True
        )
        result = self._color_match(result, original, self.last_target_face)
        if fast:
            mask = self._cached_blend_mask
            if mask is None or mask.shape[:2] != original.shape[:2]:
                return result
            blended = (
                result.astype(np.float32) * mask
                + original.astype(np.float32) * (1.0 - mask)
            )
            return blended.astype(np.uint8)
        return self._expanded_blend(original, result, self.last_target_face)

    def _color_match(self, swapped, target, face):
        """Reinhard color transfer in LAB space, applied to the face crop only.

        Shifts the swapped face's per-channel mean and std in LAB to match
        the original face crop's statistics. Result: the swapped face
        inherits the scene's lighting/skin tone instead of carrying the
        source photo's lighting onto the target frame.
        """
        h, w = swapped.shape[:2]
        bbox = face.bbox.astype(int)
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(w, int(bbox[2]))
        y2 = min(h, int(bbox[3]))
        if x2 - x1 < 8 or y2 - y1 < 8:
            return swapped

        swap_crop = swapped[y1:y2, x1:x2]
        targ_crop = target[y1:y2, x1:x2]

        s_lab = cv2.cvtColor(swap_crop, cv2.COLOR_BGR2LAB).astype(np.float32)
        t_lab = cv2.cvtColor(targ_crop, cv2.COLOR_BGR2LAB).astype(np.float32)

        for ch in range(3):
            s_mean = float(s_lab[..., ch].mean())
            s_std = float(s_lab[..., ch].std())
            t_mean = float(t_lab[..., ch].mean())
            t_std = float(t_lab[..., ch].std())
            if s_std < 1e-3:
                continue
            s_lab[..., ch] = (s_lab[..., ch] - s_mean) * (t_std / s_std) + t_mean

        s_lab = np.clip(s_lab, 0, 255).astype(np.uint8)
        out = swapped.copy()
        out[y1:y2, x1:x2] = cv2.cvtColor(s_lab, cv2.COLOR_LAB2BGR)
        return out

    def _expanded_blend(self, original, swapped, face):
        """Full-quality elliptical blend; rebuilds the mask each call."""
        mask = self._build_blend_mask(original.shape, face)
        blended = (
            swapped.astype(np.float32) * mask
            + original.astype(np.float32) * (1.0 - mask)
        )
        return blended.astype(np.uint8)

    def _build_blend_mask(self, frame_shape, face):
        """Elliptical Gaussian-feathered blend mask, expanded around the face."""
        h, w = frame_shape[:2]
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        face_w = x2 - x1
        face_h = y2 - y1

        pad_top = int(face_h * 0.45)
        pad_bottom = int(face_h * 0.1)
        pad_side = int(face_w * 0.15)

        ex1 = max(0, x1 - pad_side)
        ey1 = max(0, y1 - pad_top)
        ex2 = min(w, x2 + pad_side)
        ey2 = min(h, y2 + pad_bottom)

        mask = np.zeros((h, w), dtype=np.uint8)
        center_x = (ex1 + ex2) // 2
        center_y = (ey1 + ey2) // 2
        radius_x = max(1, (ex2 - ex1) // 2)
        radius_y = max(1, (ey2 - ey1) // 2)

        cv2.ellipse(
            mask, (center_x, center_y), (radius_x, radius_y),
            0, 0, 360, 255, -1,
        )

        blur_size = max(31, (face_w // 4) * 2 + 1)
        mask = cv2.GaussianBlur(mask, (blur_size, blur_size), blur_size // 3)

        mask_float = mask.astype(np.float32) / 255.0
        return np.stack([mask_float] * 3, axis=-1)

    def _downscale(self, frame):
        """Downscale frame for faster face detection."""
        h, w = frame.shape[:2]
        if w <= LIVE_PROCESS_WIDTH:
            return frame
        scale = LIVE_PROCESS_WIDTH / w
        new_w = LIVE_PROCESS_WIDTH
        new_h = int(h * scale)
        return cv2.resize(frame, (new_w, new_h))

    def close(self):
        pass
