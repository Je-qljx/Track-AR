import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor


class FrameTracker:
    def __init__(self, max_width: int = 640, refresh_every: int = 90):
        self.orb = cv2.ORB.create(
            nfeatures=1200,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=15,
            patchSize=31,
        )
        self.max_width = max_width
        self._first_frame = True
        # Original reference (never changes — for drift correction)
        self._first_gray: np.ndarray | None = None
        self._first_kp: list[cv2.KeyPoint] | None = None
        self._first_des: np.ndarray | None = None
        # Pairwise reference (updated every frame)
        self._ref_gray: np.ndarray | None = None
        self._ref_kp: list[cv2.KeyPoint] | None = None
        self._ref_des: np.ndarray | None = None
        self._current_H: np.ndarray = np.eye(3, dtype=np.float64)
        self._H_cumulative: np.ndarray = np.eye(3, dtype=np.float64)
        self._scale = 1.0
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._pending_feature: tuple[np.ndarray, list[cv2.KeyPoint], np.ndarray] | None = None
        # Reference refresh (for camera panning beyond initial view)
        self._refresh_every = refresh_every
        self._last_refresh = 0
        self._frame_count = 0
        self._consecutive_klt = 0  # frames where only KLT succeeds
        # KLT fallback
        self._klt_pts: np.ndarray | None = None
        self._prev_gray_lr: np.ndarray | None = None
        self._lk_win_size = (21, 21)
        self._lk_max_level = 3
        self._lk_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        self._klt_max_features = 600
        self._klt_quality = 0.005
        self._klt_min_dist = 3.0
        self._klt_min_features = 8

    @property
    def H_calib_current(self) -> np.ndarray:
        return self._current_H @ self._H_cumulative

    @H_calib_current.setter
    def H_calib_current(self, value: np.ndarray):
        self._current_H = value

    def _downscale(self, gray: np.ndarray) -> np.ndarray:
        h, w = gray.shape
        if w <= self.max_width:
            self._scale = 1.0
            return gray
        self._scale = self.max_width / w
        new_w = self.max_width
        new_h = int(h * self._scale)
        return cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    def _to_lr(self, u: float, v: float) -> tuple[float, float]:
        return u * self._scale, v * self._scale

    def _from_lr(self, u: float, v: float) -> tuple[float, float]:
        if self._scale == 0:
            return u, v
        return u / self._scale, v / self._scale

    def _compute_features(self, gray: np.ndarray) -> tuple[np.ndarray, list[cv2.KeyPoint], np.ndarray]:
        lr = self._downscale(gray)
        kp, des = self.orb.detectAndCompute(lr, None)
        return lr, kp, des

    def _detect_klt_features(self, lr: np.ndarray) -> np.ndarray | None:
        pts = cv2.goodFeaturesToTrack(
            lr, maxCorners=self._klt_max_features,
            qualityLevel=self._klt_quality,
            minDistance=self._klt_min_dist)
        return pts

    def set_reference(self, gray: np.ndarray):
        lr, kp, des = self._compute_features(gray)
        self._first_gray = lr
        self._first_kp = kp
        self._first_des = des
        self._ref_gray = lr
        self._ref_kp = kp[:] if kp else []
        self._ref_des = des.copy() if des is not None else None
        self._current_H = np.eye(3, dtype=np.float64)
        self._H_cumulative = np.eye(3, dtype=np.float64)
        self._last_refresh = 0
        self._consecutive_klt = 0
        self._klt_pts = self._detect_klt_features(lr)
        self._prev_gray_lr = lr

    def is_ready(self) -> bool:
        return self._first_gray is not None

    def need_update(self) -> bool:
        if self._first_frame:
            self._first_frame = False
            return True
        return True

    def _refresh_reference(self, lr_in: np.ndarray, kp_in: list[cv2.KeyPoint], des_in: np.ndarray):
        """Save current frame as new reference for ORB matching (rotating first_gray)."""
        self._H_cumulative = self._current_H @ self._H_cumulative
        self._first_gray = lr_in.copy()
        self._first_kp = kp_in
        self._first_des = des_in.copy()
        self._current_H = np.eye(3, dtype=np.float64)
        self._last_refresh = self._frame_count
        self._consecutive_klt = 0

    def _filter_matches(self, matches, src_kp, dst_kp, ratio_thresh: float = 0.75) -> list[cv2.DMatch]:
        good = []
        for m in matches:
            if hasattr(m, 'distance') and m.distance < ratio_thresh * 100:
                good.append(m)
        return good

    def _compute_homography(self, src_pts, dst_pts) -> tuple[np.ndarray | None, np.ndarray | None]:
        if len(src_pts) < 6:
            return None, None
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.USAC_MAGSAC, 3.0)
        if H is not None and mask is not None and int(np.sum(mask)) >= 4:
            return H, mask
        return None, None

    def _check_homography_sanity(self, H: np.ndarray, relaxed: bool = False) -> bool:
        det = np.linalg.det(H)
        if det < 0.05 or det > 10.0:
            return False
        scale = np.sqrt(det)
        if relaxed:
            if scale < 0.25 or scale > 5.0:
                return False
            if abs(H[0, 1]) > 10.0 or abs(H[1, 0]) > 10.0:
                return False
        else:
            if scale < 0.3 or scale > 5.0:
                return False
            if abs(H[0, 1]) > 5.0 or abs(H[1, 0]) > 5.0:
                return False
        return True

    def update(self, gray: np.ndarray):
        self._frame_count += 1
        if self._first_gray is None:
            self.set_reference(gray)
            return

        lr_in = self._downscale(gray)
        kp_in, des_in = self.orb.detectAndCompute(lr_in, None)
        if des_in is None or len(kp_in) < 12:
            return

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        updated = False
        info: dict[str, int | str] = {}
        was_klt = False

        # --- Reference refresh ---
        # When KLT has been running alone too long, the stored first-g frame
        # has no visual overlap with current scene. Save current frame as new
        # first-g with cumulative H preserved.
        if (self._consecutive_klt > self._refresh_every
                and (self._frame_count - self._last_refresh) > self._refresh_every // 2):
            self._refresh_reference(lr_in, kp_in, des_in)
            info['ref_refresh'] = True
            info['method'] = 'ref_refresh'
            updated = True

        # 1) Direct first-frame match against current reference (drift-free)
        if not updated and self._first_des is not None:
            m0 = bf.match(self._first_des, des_in)
            info['first_matches'] = len(m0)
            if len(m0) >= 6:
                src = np.float32([self._first_kp[m.queryIdx].pt for m in m0]).reshape(-1, 1, 2)
                dst = np.float32([kp_in[m.trainIdx].pt for m in m0]).reshape(-1, 1, 2)
                H0, mask0 = self._compute_homography(src, dst)
                n_inliers = int(np.sum(mask0)) if mask0 is not None else 0
                info['first_inliers'] = n_inliers
                if H0 is not None and n_inliers >= 10 and self._check_homography_sanity(H0, relaxed=True):
                    self._current_H = H0
                    self._H_cumulative = np.eye(3, dtype=np.float64)
                    info['method'] = 'first_frame'
                    updated = True

        # 2) Pairwise ORB fallback
        if not updated and self._ref_des is not None:
            m_p = bf.match(self._ref_des, des_in)
            info['pairwise_matches'] = len(m_p)
            if len(m_p) >= 6:
                src = np.float32([self._ref_kp[m.queryIdx].pt for m in m_p]).reshape(-1, 1, 2)
                dst = np.float32([kp_in[m.trainIdx].pt for m in m_p]).reshape(-1, 1, 2)
                H_p, mask_p = self._compute_homography(src, dst)
                info['pairwise_inliers'] = int(np.sum(mask_p)) if mask_p is not None else 0
                if H_p is not None and self._check_homography_sanity(H_p):
                    candidate = H_p @ self._current_H
                    if self._check_homography_sanity(candidate):
                        self._current_H = candidate
                        info['method'] = 'pairwise'
                        updated = True

        # 3) KLT pairwise fallback (handles motion blur / low texture)
        if not updated and self._prev_gray_lr is not None and self._klt_pts is not None:
            new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray_lr, lr_in, self._klt_pts, None,
                winSize=self._lk_win_size, maxLevel=self._lk_max_level,
                criteria=self._lk_criteria)
            if new_pts is not None and status is not None:
                good_new = new_pts[status.flatten() == 1]
                good_old = self._klt_pts[status.flatten() == 1]
                info['klt_tracked'] = len(good_new)
                info['klt_total'] = len(self._klt_pts)
                if len(good_new) >= self._klt_min_features:
                    displacement = good_new - good_old
                    dx = float(np.median(displacement[..., 0]))
                    dy = float(np.median(displacement[..., 1]))
                    if max(abs(dx), abs(dy)) < 200:
                        H_k = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=np.float64)
                        info['klt_dx'] = dx
                        info['klt_dy'] = dy
                        candidate = H_k @ self._current_H
                        if self._check_homography_sanity(candidate, relaxed=True):
                            self._current_H = candidate
                            info['method'] = 'klt'
                            updated = True
                            was_klt = True
                            self._klt_pts = good_new.reshape(-1, 1, 2)
                        else:
                            self._klt_pts = good_new.reshape(-1, 1, 2)
                    else:
                        self._klt_pts = good_new.reshape(-1, 1, 2)
                else:
                    self._klt_pts = None

        # Update consecutive KLT counter
        if was_klt:
            self._consecutive_klt += 1
        else:
            self._consecutive_klt = 0

        # 4) Periodic KLT redetect if lost
        if not updated and (self._klt_pts is None or len(self._klt_pts) < self._klt_min_features):
            if self._klt_pts is not None and len(self._klt_pts) > 0:
                extra = self._detect_klt_features(lr_in)
                if extra is not None and len(extra) > 0:
                    self._klt_pts = np.vstack([self._klt_pts, extra])
                    if len(self._klt_pts) > self._klt_max_features:
                        self._klt_pts = self._klt_pts[:self._klt_max_features]
            else:
                self._klt_pts = self._detect_klt_features(lr_in)

        if 'method' not in info:
            info['method'] = 'failed'
        self._match_info = info

        self._ref_gray = lr_in
        self._ref_kp = kp_in
        self._ref_des = des_in
        self._prev_gray_lr = lr_in

        # Keep KLT features fresh for next frame
        new_klt = self._detect_klt_features(lr_in)
        if new_klt is not None and len(new_klt) > 0:
            self._klt_pts = new_klt

    @property
    def last_match_info(self) -> dict:
        return getattr(self, '_match_info', {})

    def calib_to_current(self, u: float, v: float) -> tuple[float, float]:
        if self._first_gray is None:
            return u, v
        lu, lv = self._to_lr(u, v)
        pt = np.array([[[lu, lv]]], dtype=np.float64)
        H_total = self._current_H @ self._H_cumulative
        tr = cv2.perspectiveTransform(pt, H_total)
        return self._from_lr(float(tr[0, 0, 0]), float(tr[0, 0, 1]))

    def current_to_calib(self, u: float, v: float) -> tuple[float, float]:
        if self._first_gray is None:
            return u, v
        H_total = self._current_H @ self._H_cumulative
        H_inv = np.linalg.inv(H_total)
        lu, lv = self._to_lr(u, v)
        pt = np.array([[[lu, lv]]], dtype=np.float64)
        tr = cv2.perspectiveTransform(pt, H_inv)
        return self._from_lr(float(tr[0, 0, 0]), float(tr[0, 0, 1]))
