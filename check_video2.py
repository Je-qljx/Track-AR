import sys; sys.path.insert(0, '.')
import cv2, numpy as np
from calibration.frame_tracker import FrameTracker

path = r'D:\DJI_20250929143730_0193_D(2504csq)等2项文件\Timeline 1.mp4'
cap = cv2.VideoCapture(path)

ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
ft = FrameTracker(max_width=640, refresh_every=60)
ft.set_reference(gray)

stats = {}
total = 0
results = []
for i in range(1, min(411, 300)):
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ft.update(gray)
    info = ft.last_match_info
    method = info.get('method', '?')
    total += 1
    stats[method] = stats.get(method, 0) + 1
    if 'ref_refresh' in info:
        H = ft.H_calib_current
        print(f'  frame {i}: REFRESH, method={method}, H diag=({H[0,0]:.3f},{H[1,1]:.3f})')
    if i <= 5 or i % 30 == 0:
        fm = info.get('first_matches', 0)
        fi = info.get('first_inliers', 0)
        pm = info.get('pairwise_matches', 0)
        klt_t = info.get('klt_tracked', 0)
        print(f'  frame {i}: {method} first={fm}/{fi} pair={pm} klt={klt_t}')

print(f'\nSummary: {stats}')
print(f'Total: {total} frames')
cap.release()
