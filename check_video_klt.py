import sys; sys.path.insert(0, '.')
import cv2, numpy as np
from calibration.lane_tracker import LaneFeatureTracker

path = r'D:\DJI_20250929143730_0193_D(2504csq)等2项文件\Timeline 1.mp4'
cap = cv2.VideoCapture(path)
ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
ft = LaneFeatureTracker(max_features=600, quality_level=0.005, min_distance=3.0)
ft.set_reference(gray)

stats = {}
H_hist = []
for i in range(1, 411):
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ft.update(gray)
    info = ft.last_match_info
    method = info.get('method', '?')
    stats[method] = stats.get(method, 0) + 1
    H = ft.H_calib_current
    H_hist.append((i, H[0,0], H[1,1], H[0,2], H[1,2]))
    if info.get('klt_tracked', 0) > 0:
        print(f'  frame {i}: {method} klt_tracked={info.get("klt_tracked")} pts={info.get("total_pts")} H_diag=({H[0,0]:.4f},{H[1,1]:.4f})')

# Print first/last H
print(f'\nFirst frame H: diag=({H_hist[0][1]:.4f},{H_hist[0][2]:.4f}) trans=({H_hist[0][3]:.1f},{H_hist[0][4]:.1f})')
print(f'Last  frame H: diag=({H_hist[-1][1]:.4f},{H_hist[-1][2]:.4f}) trans=({H_hist[-1][3]:.1f},{H_hist[-1][4]:.1f})')
print(f'Summary: {stats}')
cap.release()
