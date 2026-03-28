import os
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

from config import MODELS_DIR

LIVE_PROCESS_WIDTH = 640


class FaceSwapModule:
    """High-quality face swapping using InsightFace + inswapper + expanded blend."""

    def __init__(self):
        # Face detection/analysis on CPU (CoreML has shape issues with these models)
        self.app = FaceAnalysis(
            name="buffalo_l", providers=["CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=0, det_size=(320, 320))

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
        """Detect face in frame and cache it. Returns True if found."""
        small = self._downscale(frame)
        faces = self.app.get(small)
        if faces:
            scale = frame.shape[1] / small.shape[1]
            face = faces[0]
            face.bbox = face.bbox * scale
            face.kps = face.kps * scale
            self.last_target_face = face
            return True
        return False

    def swap_with_cached_face(self, frame):
        """Swap using the last cached target face detection (fast)."""
        if self.source_face is None or self.last_target_face is None:
            return frame

        original = frame.copy()
        result = self.swapper.get(
            frame, self.last_target_face, self.source_face, paste_back=True
        )
        result = self._expanded_blend(original, result, self.last_target_face)
        return result

    def _expanded_blend(self, original, swapped, face):
        """Expand the swap region upward for better forehead/hairline blending."""
        h, w = original.shape[:2]
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox

        # Calculate face dimensions
        face_w = x2 - x1
        face_h = y2 - y1

        # Expand the region: more upward (forehead/hairline), slight sides
        pad_top = int(face_h * 0.45)
        pad_bottom = int(face_h * 0.1)
        pad_side = int(face_w * 0.15)

        # Expanded bounding box (clamped to frame)
        ex1 = max(0, x1 - pad_side)
        ey1 = max(0, y1 - pad_top)
        ex2 = min(w, x2 + pad_side)
        ey2 = min(h, y2 + pad_bottom)

        # Create an elliptical mask covering the expanded face region
        mask = np.zeros((h, w), dtype=np.uint8)
        center_x = (ex1 + ex2) // 2
        center_y = (ey1 + ey2) // 2
        radius_x = (ex2 - ex1) // 2
        radius_y = (ey2 - ey1) // 2

        cv2.ellipse(mask, (center_x, center_y), (radius_x, radius_y),
                    0, 0, 360, 255, -1)

        # Heavy Gaussian blur for smooth feathered edges
        blur_size = max(31, (face_w // 4) * 2 + 1)
        mask = cv2.GaussianBlur(mask, (blur_size, blur_size), blur_size // 3)

        # Blend: swapped face in mask region, original outside
        mask_float = mask.astype(np.float32) / 255.0
        mask_3ch = np.stack([mask_float] * 3, axis=-1)

        blended = (swapped.astype(np.float32) * mask_3ch +
                   original.astype(np.float32) * (1.0 - mask_3ch))

        return blended.astype(np.uint8)

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
