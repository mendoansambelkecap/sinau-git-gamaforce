"""
AntiJomok - Real-time Hand Gesture Filter Pipeline
"""

from dataclasses import dataclass
from pathlib import Path
import random
import time
from typing import Dict, List, Optional, Tuple, Callable

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class PipelineConfig:
    cam_index: int = 0
    frame_width: int = 960
    frame_height: int = 540
    pinch_threshold_px: float = 45.0
    filter_cooldown_sec: float = 0.15
    mode_cooldown_sec: float = 1.2
    fist_dist_threshold_px: float = 80.0
    meme_dir: Path = Path(__file__).resolve().parent / "memes"
    meme_max_side_px: int = 300
    meme_hold_sec: float = 0.25    # gesture must be held this long to trigger
    meme_linger_sec: float = 1.2   # meme stays this long after the gesture ends


# gesture -> (meme file, HUD label)
GESTURE_MEMES: Dict[str, Tuple[str, str]] = {
    "thumbs_up": ("jempol.jpg", "JEMPOL"),
    "middle": ("jari_tengah.jpg", "JARI TENGAH"),
    "peace": ("anak_lelaki.jpg", "PEACE"),
    "point": ("izin_berpikir.jpg", "NUNJUK"),
    "ok": ("nggakapaapa.jpg", "OK"),
    "rock": ("son.jpg", "METAL"),
    "two_open": ("sopo_iki_cik.jpg", "2 TANGAN TERBUKA"),
    "two_fist": ("aku_ngerti.jpg", "2 TANGAN KEPAL"),
    "covering": ("takut.jpg", "TUTUP MUKA"),
}


class FilterBank:
    @staticmethod
    def dual_tone(roi: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)
        out = np.zeros_like(roi)
        out[mask == 255] = (10, 140, 255)
        out[mask == 0] = (180, 30, 220)
        return out

    @staticmethod
    def thermal(roi: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    @staticmethod
    def sketch(roi: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        inv = 255 - gray
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def pixelate(roi: np.ndarray, block_size: int = 14) -> np.ndarray:
        h, w = roi.shape[:2]
        if h < 2 or w < 2:
            return roi
        small = cv2.resize(roi, (max(1, w // block_size), max(1, h // block_size)), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def glitch(roi: np.ndarray) -> np.ndarray:
        h, w = roi.shape[:2]
        if h < 2 or w < 2:
            return roi
        b, g, r = cv2.split(roi)
        shift = random.randint(4, 12)
        r = np.roll(r, shift, axis=1)
        b = np.roll(b, -shift, axis=1)
        out = cv2.merge([b, g, r])
        for _ in range(2):
            y = random.randint(0, h - 1)
            out[y : y + 1, :] = np.random.randint(0, 255, (1, w, 3), dtype=np.uint8)
        return out

    @staticmethod
    def invert(roi: np.ndarray) -> np.ndarray:
        return 255 - roi

    @staticmethod
    def red_channel(roi: np.ndarray) -> np.ndarray:
        b, g, r = cv2.split(roi)
        zeros = np.zeros_like(b)
        return cv2.merge([zeros, zeros, r])

    @staticmethod
    def edge(roi: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 150)
        colored = cv2.applyColorMap(edges, cv2.COLORMAP_SUMMER)
        return cv2.bitwise_and(colored, colored, mask=edges)

    @staticmethod
    def blur(roi: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(roi, (25, 25), 0)

    @staticmethod
    def cartoon(roi: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(roi, 9, 250, 250)
        return cv2.bitwise_and(color, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))

    @staticmethod
    def rainbow_wave(roi: np.ndarray) -> np.ndarray:
        h, w = roi.shape[:2]
        t = time.time() * 5.0
        x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
        pattern = np.sin((x_coords + y_coords) * 0.05 + t) * 127 + 128
        rainbow = cv2.applyColorMap(pattern.astype(np.uint8), cv2.COLORMAP_HSV)
        return cv2.addWeighted(roi, 0.3, rainbow, 0.7, 0)


class GeometryUtils:
    @staticmethod
    def euclidean_dist(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))

    @staticmethod
    def is_fist_closed(landmarks, w: int, h: int, threshold: float) -> bool:
        wrist = np.array([landmarks[0].x * w, landmarks[0].y * h])
        tips = [8, 12, 16, 20]
        distances = [np.linalg.norm(np.array([landmarks[t].x * w, landmarks[t].y * h]) - wrist) for t in tips]
        return float(np.mean(distances)) < threshold

    @staticmethod
    def is_hand_rotated(thumb: Tuple[int, int], index: Tuple[int, int]) -> bool:
        dx, dy = index[0] - thumb[0], index[1] - thumb[1]
        return (dy > 25) or (abs(dx) > abs(dy) * 1.1)

    @staticmethod
    def sort_quad_clean(pts: List[Tuple[int, int]]) -> np.ndarray:
        arr = np.array(pts, dtype=np.float32)
        x_sorted = arr[np.argsort(arr[:, 0]), :]
        leftmost = x_sorted[:2, :][np.argsort(x_sorted[:2, 1]), :]
        rightmost = x_sorted[2:, :][np.argsort(x_sorted[2:, 1]), :]
        return np.array([leftmost[0], rightmost[0], rightmost[1], leftmost[1]], dtype=np.int32)

    @staticmethod
    def sort_quad_bowtie(pts: List[Tuple[int, int]]) -> np.ndarray:
        arr = np.array(pts, dtype=np.float32)
        x_sorted = arr[np.argsort(arr[:, 0]), :]
        leftmost = x_sorted[:2, :][np.argsort(x_sorted[:2, 1]), :]
        rightmost = x_sorted[2:, :][np.argsort(x_sorted[2:, 1]), :]
        return np.array([leftmost[0], rightmost[1], rightmost[0], leftmost[1]], dtype=np.int32)


class GestureDetector:
    FINGERS = [(8, 6), (12, 10), (16, 14), (20, 18)]  # (tip, pip): index, middle, ring, pinky

    @staticmethod
    def points(landmarks, w: int, h: int) -> np.ndarray:
        return np.array([[p.x * w, p.y * h] for p in landmarks])

    @staticmethod
    def palm_size(pts: np.ndarray) -> float:
        return float(np.linalg.norm(pts[9] - pts[0])) + 1e-6

    @staticmethod
    def classify(pts: np.ndarray) -> Optional[str]:
        d = lambda a, b: float(np.linalg.norm(pts[a] - pts[b]))
        palm = GestureDetector.palm_size(pts)
        # a finger is extended when its tip is clearly farther from the wrist than its middle joint
        idx, mid, ring, pinky = [d(tip, 0) > d(pip, 0) * 1.15 for tip, pip in GestureDetector.FINGERS]
        thumb = d(4, 17) > d(3, 17) and d(4, 17) > 0.9 * palm

        if d(4, 8) < 0.3 * palm and mid and ring and pinky:
            return "ok"
        if not (idx or mid or ring or pinky):
            if thumb and pts[4][1] < pts[5][1] - 0.2 * palm:
                return "thumbs_up"
            return "fist"
        if idx and mid and ring and pinky:
            return "open" if thumb else None
        states = (idx, mid, ring, pinky)
        return {
            (False, True, False, False): "middle",
            (True, True, False, False): "peace",
            (True, False, False, False): "point",
            (True, False, False, True): "rock",
        }.get(states)

    @staticmethod
    def classify_pair(a: np.ndarray, b: np.ndarray) -> Optional[str]:
        palm = (GestureDetector.palm_size(a) + GestureDetector.palm_size(b)) / 2
        if np.linalg.norm(a[9] - b[9]) < 1.0 * palm:
            return "covering"
        ga, gb = GestureDetector.classify(a), GestureDetector.classify(b)
        if ga == gb == "open":
            return "two_open"
        if ga == gb == "fist":
            return "two_fist"
        return None


class MemeReactor:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.memes: Dict[str, np.ndarray] = {}
        for gesture, (fname, _) in GESTURE_MEMES.items():
            img = cv2.imread(str(cfg.meme_dir / fname))
            if img is None:
                print(f"[WARN] Meme tidak ketemu: {cfg.meme_dir / fname}")
                continue
            scale = cfg.meme_max_side_px / max(img.shape[:2])
            self.memes[gesture] = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        self.candidate: Optional[str] = None
        self.candidate_since = 0.0
        self.active: Optional[str] = None
        self.shown_at = 0.0
        self.last_seen = 0.0
        self.anchor = (0, 0)

    def update(self, gesture: Optional[str], anchor: Tuple[int, int], now: float) -> None:
        if gesture != self.candidate:
            self.candidate, self.candidate_since = gesture, now
        if gesture in self.memes and now - self.candidate_since >= self.cfg.meme_hold_sec:
            if gesture != self.active or now - self.last_seen > self.cfg.meme_linger_sec:
                self.active, self.shown_at = gesture, now
            self.last_seen, self.anchor = now, anchor

    def draw(self, frame: np.ndarray, now: float) -> None:
        if self.active is None:
            return
        remaining = self.cfg.meme_linger_sec - (now - self.last_seen)
        if remaining <= 0:
            self.active = None
            return

        pop = min(1.0, (now - self.shown_at) / 0.18)
        scale = 0.6 + 0.4 * (1 - (1 - pop) ** 3)
        alpha = min(1.0, remaining / 0.3)

        img = cv2.resize(self.memes[self.active], None, fx=scale, fy=scale)
        h, w = img.shape[:2]
        fh, fw = frame.shape[:2]
        x = int(np.clip(self.anchor[0] - w / 2, 0, fw - w))
        y = int(np.clip(self.anchor[1] - h - 20, 0, fh - h))

        roi = frame[y : y + h, x : x + w]
        frame[y : y + h, x : x + w] = cv2.addWeighted(img, alpha, roi, 1 - alpha, 0)
        cv2.rectangle(frame, (x, y), (x + w - 1, y + h - 1), (255, 255, 255), 3)


class PortalProcessor:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.filters: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
            "dual-tone": FilterBank.dual_tone,
            "thermal": FilterBank.thermal,
            "sketch": FilterBank.sketch,
            "pixelate": FilterBank.pixelate,
            "glitch": FilterBank.glitch,
            "invert": FilterBank.invert,
            "red-channel": FilterBank.red_channel,
            "edge": FilterBank.edge,
            "blur": FilterBank.blur,
            "cartoon": FilterBank.cartoon,
            "rainbow-wave": FilterBank.rainbow_wave,
        }
        self.filter_keys = list(self.filters.keys())
        self.active_filter_idx = 0
        self.is_3d_mode = False
        self.meme_mode = True
        self.meme_reactor = MemeReactor(cfg)
        self.current_gesture: Optional[str] = None

        self.last_switch_time = 0.0
        self.last_mode_toggle = 0.0

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.detector = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8,
        )

    @property
    def current_filter_name(self) -> str:
        return self.filter_keys[self.active_filter_idx]

    @property
    def secondary_filter_name(self) -> str:
        return self.filter_keys[(self.active_filter_idx + 1) % len(self.filter_keys)]

    def cycle_filter(self, step: int = 1) -> None:
        self.active_filter_idx = (self.active_filter_idx + step) % len(self.filter_keys)

    def render_portal(self, frame: np.ndarray, pts: List[Tuple[int, int]], filter_key: str) -> np.ndarray:
        poly = np.array(pts, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(poly)
        x, y = max(0, x), max(0, y)
        w, h = min(w, frame.shape[1] - x), min(h, frame.shape[0] - y)

        if w <= 10 or h <= 10:
            return frame

        roi = frame[y : y + h, x : x + w].copy()
        processed_roi = self.filters[filter_key](roi)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly - [x, y]], 255)
        mask_3c = cv2.merge([mask, mask, mask])

        bg = cv2.bitwise_and(roi, cv2.bitwise_not(mask_3c))
        fg = cv2.bitwise_and(processed_roi, mask_3c)
        frame[y : y + h, x : x + w] = cv2.add(bg, fg)

        cv2.polylines(frame, [poly], isClosed=True, color=(255, 255, 255), thickness=2)
        return frame

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (self.cfg.frame_width, self.cfg.frame_height))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = self.detector.process(rgb)
        now = time.time()

        if self.meme_mode:
            self._process_memes(frame, results, now)
            self._draw_hud(frame, False)
            return frame

        all_hand_tips = []
        fist_count = 0
        is_bowtie = False

        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_lm, self.mp_hands.HAND_CONNECTIONS)
                
                lm = hand_lm.landmark
                tips = [(int(lm[i].x * self.cfg.frame_width), int(lm[i].y * self.cfg.frame_height)) for i in [4, 8, 12, 16, 20]]
                all_hand_tips.append(tips)

                # Fast Pinch
                if GeometryUtils.euclidean_dist(tips[0], tips[4]) < self.cfg.pinch_threshold_px:
                    if now - self.last_switch_time > self.cfg.filter_cooldown_sec:
                        self.cycle_filter(1)
                        self.last_switch_time = now

                if GeometryUtils.is_fist_closed(lm, self.cfg.frame_width, self.cfg.frame_height, self.cfg.fist_dist_threshold_px):
                    fist_count += 1

            # Dual Fist Mode Switch
            if fist_count == 2 and (now - self.last_mode_toggle > self.cfg.mode_cooldown_sec):
                self.is_3d_mode = not self.is_3d_mode
                self.last_mode_toggle = now

            if self.is_3d_mode:
                if len(all_hand_tips) == 2:
                    t1, t2 = all_hand_tips[0], all_hand_tips[1]
                    frame = self.render_portal(frame, [t1[0], t1[1], t1[2], t2[2], t2[1], t2[0]], self.current_filter_name)
                    frame = self.render_portal(frame, [t1[2], t1[3], t1[4], t2[4], t2[3], t2[2]], self.secondary_filter_name)
                elif len(all_hand_tips) == 1:
                    frame = self.render_portal(frame, all_hand_tips[0], self.current_filter_name)
            else:
                if len(all_hand_tips) == 2:
                    corners = [all_hand_tips[0][0], all_hand_tips[0][1], all_hand_tips[1][0], all_hand_tips[1][1]]
                    if GeometryUtils.is_hand_rotated(corners[0], corners[1]) or GeometryUtils.is_hand_rotated(corners[2], corners[3]):
                        quad = GeometryUtils.sort_quad_bowtie(corners)
                        is_bowtie = True
                    else:
                        quad = GeometryUtils.sort_quad_clean(corners)
                    frame = self.render_portal(frame, quad, self.current_filter_name)
                elif len(all_hand_tips) == 1:
                    t = all_hand_tips[0]
                    frame = self.render_portal(frame, [t[0], t[1], t[2], t[4]], self.current_filter_name)

        self._draw_hud(frame, is_bowtie)
        return frame

    def _process_memes(self, frame: np.ndarray, results, now: float) -> None:
        hands = []
        for hand_lm in results.multi_hand_landmarks or []:
            self.mp_draw.draw_landmarks(frame, hand_lm, self.mp_hands.HAND_CONNECTIONS)
            hands.append(GestureDetector.points(hand_lm.landmark, self.cfg.frame_width, self.cfg.frame_height))

        gesture, anchor = None, (0, 0)
        if len(hands) == 2:
            gesture = GestureDetector.classify_pair(hands[0], hands[1])
            if gesture:
                both = np.vstack(hands)
                anchor = (int(both[:, 0].mean()), int(both[:, 1].min()))
        if gesture is None:
            for pts in hands:
                g = GestureDetector.classify(pts)
                if g in GESTURE_MEMES:
                    gesture, anchor = g, (int(pts[:, 0].mean()), int(pts[:, 1].min()))
                    break

        self.current_gesture = gesture
        self.meme_reactor.update(gesture, anchor, now)
        self.meme_reactor.draw(frame, now)

    def _draw_gesture_legend(self, frame: np.ndarray) -> None:
        rows = [(g, f"{label} -> {fname.rsplit('.', 1)[0].replace('_', ' ')}") for g, (fname, label) in GESTURE_MEMES.items()]
        line_h, pad = 20, 8
        box_w = 300
        box_h = line_h * (len(rows) + 1) + pad
        x0, y0 = frame.shape[1] - box_w - 10, frame.shape[0] - box_h - 10

        panel = frame[y0 : y0 + box_h, x0 : x0 + box_w]
        frame[y0 : y0 + box_h, x0 : x0 + box_w] = cv2.addWeighted(panel, 0.4, np.zeros_like(panel), 0.6, 0)
        cv2.putText(frame, "DAFTAR GESTURE", (x0 + pad, y0 + line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        for i, (gesture, text) in enumerate(rows, start=2):
            color = (0, 255, 0) if gesture == self.current_gesture else (230, 230, 230)
            cv2.putText(frame, text, (x0 + pad, y0 + line_h * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    def _draw_hud(self, frame: np.ndarray, is_bowtie: bool) -> None:
        if self.meme_mode:
            label = GESTURE_MEMES[self.current_gesture][1] if self.current_gesture else "-"
            cv2.putText(frame, "MODE: MEME [Key 'M' = Portal]", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            cv2.putText(frame, f"GESTURE: {label}", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            self._draw_gesture_legend(frame)
            return
        mode_str = "3D Mesh" if self.is_3d_mode else ("2D Bowtie" if is_bowtie else "2D Quad")
        cv2.putText(frame, f"MODE: {mode_str} [Key 'C' / Dual Fist, 'M' = Meme]", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(frame, f"FILTER: {self.current_filter_name.upper()} [Pinch / Key 'N'/'P']", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def main() -> None:
    cfg = PipelineConfig()
    processor = PortalProcessor(cfg)
    cap = cv2.VideoCapture(cfg.cam_index)

    if not cap.isOpened():
        print("[ERROR] Kamera tidak terdeteksi!")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Gagal membaca frame kamera.")
            break

        out_frame = processor.process_frame(frame)
        cv2.imshow("AntiJomok", out_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("m"):
            processor.meme_mode = not processor.meme_mode
        elif key == ord("c"):
            processor.is_3d_mode = not processor.is_3d_mode
        elif key == ord("n"):
            processor.cycle_filter(1)
        elif key == ord("p"):
            processor.cycle_filter(-1)
        elif key == ord("s"):
            cv2.imwrite(f"cap_{int(time.time())}.png", out_frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()