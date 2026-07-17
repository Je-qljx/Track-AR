import sys; sys.path.insert(0, '.')
import cv2
import numpy as np

path = r"D:\DJI_20250929143730_0193_D(2504csq)等2项文件\Timeline 1.mp4"
cap = cv2.VideoCapture(path)
if not cap.isOpened():
    print(f"FAILED to open: {path}")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"FPS: {fps:.2f}, Size: {w}x{h}, Frames: {total}, Duration: {total/fps:.1f}s")

# Check first few frames
for i in range(5):
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = gray.mean()
    std_brightness = gray.std()
    # Count ORB features
    orb = cv2.ORB.create(nfeatures=500)
    kp = orb.detect(gray, None)
    print(f"  Frame {i}: mean={mean_brightness:.1f} std={std_brightness:.1f} ORB_kp={len(kp)}")

# Check for track features (goodFeaturesToTrack)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gftt = cv2.goodFeaturesToTrack(gray, maxCorners=500, qualityLevel=0.01, minDistance=5)
print(f"  goodFeaturesToTrack: {len(gftt) if gftt is not None else 0}")

# Sample every 100 frames to see camera motion
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
prev_gray = None
motion_sum = 0
motion_count = 0
for i in range(0, min(total, 600), 30):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if prev_gray is not None:
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        motion_sum += mag.mean()
        motion_count += 1
    prev_gray = gray

if motion_count > 0:
    print(f"  Mean optical flow: {motion_sum/motion_count:.2f} px/frame (every 30 frames)")

# Check if video has track lines visible
cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
ret, frame = cap.read()
if ret:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)
    n_lines = len(lines) if lines is not None else 0
    print(f"  HoughLinesP at mid-frame: {n_lines}")

cap.release()
print("Done")
