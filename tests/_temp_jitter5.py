import sys; sys.path.insert(0, '.')
import numpy as np, cv2, math
from calibration.coords import TrackGeometry, WorldCoord, ImageCoord
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
track_pts = list(pts) + [WorldCoord(dm, geom.lane_center_y(l), 0.0) for l in range(1,9) for dm in range(0,101,10)]
pipeline.projector.set_calibration_world_pts(track_pts)

ft = pipeline.frame_tracker

render_proj = Projector(K, np.zeros((4,1)))
render_proj.set_extrinsics(r0.copy(), t0.copy())
scene = SyntheticScene(render_proj, geom, speeds=[9.5]*8)
fps = 60.0

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

# Track first few frames and check first_pts
for fi in range(65):
    rvec, tvec = perturb_jitter(fi, fps, r0, t0)
    render_proj.set_extrinsics(rvec, tvec)
    canvas = scene.render_background(scene.update(fi/fps))
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    ft.update(gray)
    mi = ft.last_match_info
    if ft._first_pts is not None:
        print(f'fi={fi}: first_pts={len(ft._first_pts)}, first_gray={ft._first_gray.shape if ft._first_gray is not None else None}, frame_count={ft._frame_count}, method={mi.get("method","?")}')
        if fi in [60, 61, 62, 63, 64]:
            first_curr, first_ref = ft._track_klt_to_first(gray)
            print(f'  first_tracked={len(first_curr)}')
