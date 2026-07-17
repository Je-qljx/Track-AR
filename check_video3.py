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
for i in range(1, 411):
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ft.update(gray)
    info = ft.last_match_info
    method = info.get('method', '?')
    stats[method] = stats.get(method, 0) + 1
    if method != 'first_frame':
        print(f'frame {i}: {method}')

print(f'Summary: {stats}')
H_end = ft.H_calib_current
print(f'H diag at end: ({H_end[0,0]:.4f}, {H_end[1,1]:.4f})')
cap.release()
