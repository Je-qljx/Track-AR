import sys; sys.path.insert(0, '.')
import numpy as np, cv2, math
from calibration.coords import TrackGeometry, ImageCoord
from calibration.projector import Projector
from tests.synthetic_scene import SyntheticScene
from calibration.frame_tracker import FrameTracker

K = np.array([[700,0,960],[0,700,540],[0,0,1]], dtype=np.float64)
r0 = np.array([[-1.9865],[-0.7462],[-0.4312]], dtype=np.float64)
t0 = np.array([[-40.827],[-14.574],[36.127]], dtype=np.float64)
geom = TrackGeometry(track_type='100m')
render_proj = Projector(K, np.zeros((4,1)))
render_proj.set_extrinsics(r0.copy(), t0.copy())
scene = SyntheticScene(render_proj, geom, speeds=[9.5]*8)

def perturb_jitter(f, fps, r0, t0):
    r = r0.copy()
    t = t0.copy()
    r[1, 0] += 0.0003 * math.sin(2*math.pi*f/(fps*10))
    rng = np.random.RandomState(int(f))
    r[0, 0] += rng.uniform(-0.00015, 0.00015)
    r[2, 0] += rng.uniform(-0.00015, 0.00015)
    t[1, 0] += rng.uniform(-0.05, 0.05)
    t[2, 0] += rng.uniform(-0.05, 0.05)
    return r, t

fps = 60.0
ft = FrameTracker(max_width=640, max_features=600)

# Frame 0: set reference
canvas0 = scene.render_background([])
gray0 = cv2.cvtColor(canvas0, cv2.COLOR_BGR2GRAY)
ft.update(gray0)
print(f'Frame 0: first_pts={ft._first_pts.shape if ft._first_pts is not None else None}, '
      f'prev_pts={ft._prev_pts.shape if ft._prev_pts is not None else None}')

# Frames 1-5: track with jitter
test_pt = np.array([[[960.0 * ft._scale, 540.0 * ft._scale]]], dtype=np.float64)
for fi in range(1, 6):
    rvec, tvec = perturb_jitter(fi, fps, r0, t0)
    render_proj.set_extrinsics(rvec, tvec)
    canvas = scene.render_background([])
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    ft.update(gray)
    mi = ft.last_match_info
    H = ft.H_calib_current
    disp = cv2.perspectiveTransform(test_pt, H) - test_pt
    print(f'Frame {fi}: method={mi.get("method","?")}, '
          f'first_pts={mi.get("first_pts_count",0)}, '
          f'first_tracked={mi.get("first_tracked",0)}, '
          f'H_disp={np.linalg.norm(disp[0,0]):.4f}px')
