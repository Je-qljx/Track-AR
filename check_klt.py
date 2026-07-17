import sys; sys.path.insert(0, '.')
import cv2, numpy as np
from calibration.frame_tracker import FrameTracker

path = r'D:\DJI_20250929143730_0193_D(2504csq)等2项文件\Timeline 1.mp4'
cap = cv2.VideoCapture(path)
ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

ft = FrameTracker(max_width=640)
ft.set_reference(gray)

for i in range(1, 145):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ft.update(gray)
    info = ft.last_match_info
    method = info.get('method', '?')
    if method == 'failed':
        # Manually compute KLT H to check
        lr = ft._downscale(gray)
        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            ft._prev_gray_lr, lr, ft._klt_pts, None,
            winSize=ft._lk_win_size, maxLevel=ft._lk_max_level,
            criteria=ft._lk_criteria)
        if new_pts is not None and status is not None:
            good_new = new_pts[status.flatten() == 1]
            good_old = ft._klt_pts[status.flatten() == 1]
            n = len(good_new)
            if n >= 8:
                Hk, mask = cv2.findHomography(good_old, good_new, cv2.USAC_MAGSAC, 3.0)
                if Hk is not None and mask is not None:
                    inl = int(np.sum(mask))
                    det = np.linalg.det(Hk)
                    scale = np.sqrt(abs(det))
                    print(f'Frame {i}: KLT tracked={n}/{len(ft._klt_pts)} inliers={inl} det={det:.4f} scale={scale:.4f} H[0,1]={Hk[0,1]:.4f} H[0,2]={Hk[0,2]:.1f} H[1,2]={Hk[1,2]:.1f}')
        break

cap.release()
