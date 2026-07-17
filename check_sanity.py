import sys; sys.path.insert(0, '.')
import cv2, numpy as np
from calibration.frame_tracker import FrameTracker

path = r'D:\DJI_20250929143730_0193_D(2504csq)等2项文件\Timeline 1.mp4'
cap = cv2.VideoCapture(path)
ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

ft = FrameTracker(max_width=640)
ft.set_reference(gray)
first_kp = ft._first_kp
first_des = ft._first_des

for target_frame in [130, 150, 180]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lr = ft._downscale(gray)
    kp, des = ft.orb.detectAndCompute(lr, None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    m0 = bf.match(first_des, des)
    print(f'\nFrame {target_frame}: {len(m0)} matches')
    
    if len(m0) >= 6:
        src = np.float32([first_kp[m.queryIdx].pt for m in m0]).reshape(-1, 1, 2)
        dst = np.float32([kp[m.trainIdx].pt for m in m0]).reshape(-1, 1, 2)
        
        # MAGSAC
        Hm, maskm = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 3.0)
        inliers_m = int(np.sum(maskm)) if maskm is not None else 0
        print(f'  MAGSAC: H={Hm is not None}, inliers={inliers_m}')
        if Hm is not None:
            det = np.linalg.det(Hm)
            scale = np.sqrt(abs(det))
            print(f'  H: det={det:.4f} scale={scale:.4f}')
            print(f'  H[0,1]={Hm[0,1]:.4f} H[1,0]={Hm[1,0]:.4f}')
            print(f'  H[0,2]={Hm[0,2]:.1f} H[1,2]={Hm[1,2]:.1f}')
            print(f'  H[2,0]={Hm[2,0]:.6f} H[2,1]={Hm[2,1]:.6f}')
            print(f'  sanity={"PASS" if ft._check_homography_sanity(Hm) else "FAIL"}')
            
            # What if we relax shear?
            det2 = np.linalg.det(Hm)
            scale2 = np.sqrt(abs(det2))
            shear_ok = abs(Hm[0,1]) < 5.0 and abs(Hm[1,0]) < 5.0
            det_ok = abs(det2) > 0.1 and abs(det2) < 10.0
            scale_ok = scale2 > 0.3 and scale2 < 5.0
            print(f'  relaxed: shear={shear_ok} det={det_ok} scale={scale_ok}')
        
        # LMEDS
        Hl, _ = cv2.findHomography(src, dst, cv2.LMEDS)
        print(f'  LMEDS: H={Hl is not None}')
        if Hl is not None:
            det = np.linalg.det(Hl)
            scale = np.sqrt(abs(det))
            print(f'  H: det={det:.4f} scale={scale:.4f}')
            print(f'  H[0,1]={Hl[0,1]:.4f} H[1,0]={Hl[1,0]:.4f}')
            print(f'  H[0,2]={Hl[0,2]:.1f} H[1,2]={Hl[1,2]:.1f}')
            print(f'  H[2,0]={Hl[2,0]:.6f} H[2,1]={Hl[2,1]:.6f}')
            print(f'  sanity={"PASS" if ft._check_homography_sanity(Hl) else "FAIL"}')

cap.release()
