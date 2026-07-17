import sys; sys.path.insert(0, '.')
import numpy as np, cv2
from calibration.coords import TrackGeometry, ImageCoord
from calibration.projector import Projector
from pipeline.main_pipeline import TrackARPipeline
from tests.synthetic_scene import SyntheticScene

K = np.array([[700,0,960],[0,700,540],[0,0,1]], dtype=np.float64)
r0 = np.array([[-1.9865],[-0.7462],[-0.4312]], dtype=np.float64)
t0 = np.array([[-40.827],[-14.574],[36.127]], dtype=np.float64)
geom = TrackGeometry(track_type='100m')
pipeline = TrackARPipeline(camera_matrix=K, geometry=geom)
pts = geom.calibration_world_points()
wa = np.array([w.as_array for w in pts], dtype=np.float64)
pj, _ = cv2.projectPoints(wa, r0, t0, K, np.zeros((4,1)))
ip = [ImageCoord(float(p[0,0]), float(p[0,1])) for p in pj]
pipeline.calibrate_from_points(pts, ip)
track_pts = list(pts) + geom.build_tracking_grid(step=10.0)
pipeline.projector.set_calibration_world_pts(track_pts)
render_proj = Projector(K, np.zeros((4,1)))
render_proj.set_extrinsics(r0.copy(), t0.copy())
scene = SyntheticScene(render_proj, geom, speeds=[9.5]*8)

# First call: set reference
canvas0 = scene.render_background([])
gray0 = cv2.cvtColor(canvas0, cv2.COLOR_BGR2GRAY)
ft = pipeline.frame_tracker
ft.update(gray0)
print(f'After first update: first_pts={ft._first_pts.shape if ft._first_pts is not None else None}')
print(f'  prev_pts={ft._prev_pts.shape if ft._prev_pts is not None else None}')

# Second frame (pose with jitter)
import math
r0_ = r0.copy(); t0_ = t0.copy()
r0_[1,0] += 0.0003 * math.sin(2*math.pi*1/(60*10))
rng = np.random.RandomState(1)
r0_[0,0] += rng.uniform(-0.00015, 0.00015)
r0_[2,0] += rng.uniform(-0.00015, 0.00015)
t0_[1,0] += rng.uniform(-0.05, 0.05)
t0_[2,0] += rng.uniform(-0.05, 0.05)
render_proj.set_extrinsics(r0_, t0_)

canvas1 = scene.render_background([])
gray1 = cv2.cvtColor(canvas1, cv2.COLOR_BGR2GRAY)

# Track KLT manually
lr0 = ft._downscale(gray0)
lr1 = ft._downscale(gray1)
pts0 = ft._first_pts
curr, ref = ft._track_klt(lr0, lr1, pts0)
print(f'KLT tracking (frame0->frame1): tracked {len(curr)}/{len(pts0)}')
if len(curr) >= 8:
    H, mask = cv2.findHomography(ref, curr, cv2.USAC_MAGSAC, 3.0)
    print(f'  Homography inliers: {int(np.sum(mask)) if mask is not None else 0}')
    if H is not None and mask is not None:
        tst = np.array([[[960*ft._scale, 540*ft._scale]]], dtype=np.float64)
        disp = cv2.perspectiveTransform(tst, H) - tst
        print(f'  H displacement at center: {np.linalg.norm(disp[0,0]):.3f}px')

# Now actually run the full update through process_frame
ft2 = FrameTracker(max_width=640)  # fresh tracker
try:
    from calibration.frame_tracker import FrameTracker
except ImportError:
    pass

# Actually let me just process the frame through pipeline
print('\nNow running through pipeline...')
pipeline2 = TrackARPipeline(camera_matrix=K, geometry=geom)
pipeline2.calibrate_from_points(pts, ip)
pipeline2.projector.set_calibration_world_pts(track_pts)
render_proj2 = Projector(K, np.zeros((4,1)))
render_proj2.set_extrinsics(r0.copy(), t0.copy())
scene2 = SyntheticScene(render_proj2, geom, speeds=[9.5]*8)

# Process frame 0
canvas0b = scene2.render_background([])
pipeline2.process_frame(canvas0b, timestamp=0.0, external_detections=[])
ft2 = pipeline2.frame_tracker
print(f'After process_frame frame 0: first_pts={ft2._first_pts.shape if ft2._first_pts is not None else None}')
print(f'  prev_pts={ft2._prev_pts.shape if ft2._prev_pts is not None else None}')
print(f'  frame_counter={ft2._frame_count}')

# Manually call update with gray1
ft2.update(gray1)
print(f'After manual update frame 1: first_pts={ft2._first_pts.shape if ft2._first_pts is not None else None}')
print(f'  prev_pts={ft2._prev_pts.shape if ft2._prev_pts is not None else None}')
print(f'  match_info={ft2._match_info}')
