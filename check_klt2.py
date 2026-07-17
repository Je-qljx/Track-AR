import sys; sys.path.insert(0, '.')
import cv2, numpy as np
from calibration.frame_tracker import FrameTracker

path = r'D:\DJI_20250929143730_0193_D(2504csq)等2项文件\Timeline 1.mp4'
cap = cv2.VideoCapture(path)
ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

ft = FrameTracker(max_width=640)
ft.set_reference(gray)

# Process up to a failed frame
target = 142
for i in range(1, target + 1):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ft.update(gray)
    info = ft.last_match_info
    if i == target:
        print(f'Frame {i}: method={info.get("method")}')
        print(f'  first_matches={info.get("first_matches")} first_inliers={info.get("first_inliers")}')
        print(f'  pairwise_matches={info.get("pairwise_matches")} pairwise_inliers={info.get("pairwise_inliers")}')
        print(f'  klt_tracked={info.get("klt_tracked")} klt_inliers={info.get("klt_inliers")}')
        print(f'  klt_total={info.get("klt_total")}')
        
        # Manually verify KLT H sanity
        lr = ft._downscale(gray)
        if ft._prev_gray_lr is not None and ft._klt_pts is not None and len(ft._klt_pts) >= 8:
            new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                ft._prev_gray_lr, lr, ft._klt_pts, None,
                winSize=ft._lk_win_size, maxLevel=ft._lk_max_level,
                criteria=ft._lk_criteria)
            if new_pts is not None and status is not None:
                good_new = new_pts[status.flatten() == 1]
                good_old = ft._klt_pts[status.flatten() == 1]
                Hk, maskk = cv2.findHomography(good_old, good_new, cv2.USAC_MAGSAC, 3.0)
                if Hk is not None and maskk is not None:
                    det = np.linalg.det(Hk)
                    inl = int(np.sum(maskk))
                    print(f'  KLT H: det={det:.6f} inliers={inl}')
                    print(f'  KLT sanity normal: {ft._check_homography_sanity(Hk)}')
                    print(f'  KLT sanity relaxed: {ft._check_homography_sanity(Hk, relaxed=True)}')
                    print(f'  Hk[0,1]={Hk[0,1]:.4f} Hk[1,0]={Hk[1,0]:.4f}')
                    print(f'  Hk[0,2]={Hk[0,2]:.1f} Hk[1,2]={Hk[1,2]:.1f}')
                    
                    # Candidate check
                    cand = Hk @ ft.H_calib_current
                    print(f'  Candidate sanity normal: {ft._check_homography_sanity(cand)}')
                    print(f'  Candidate sanity relaxed: {ft._check_homography_sanity(cand, relaxed=True)}')
                    det_c = np.linalg.det(cand)
                    print(f'  Candidate det={det_c:.6f}')

cap.release()
