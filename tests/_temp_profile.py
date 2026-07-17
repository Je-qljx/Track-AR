import sys; sys.path.insert(0, '.')
import numpy as np, cv2, math, time
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

fps = 60.0
pipeline.running = True

# Profile frame_tracker alone
for fi in range(5):
    canvas = scene.render_background(scene.update(fi/fps))
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    if fi == 0:
        pipeline.frame_tracker.update(gray)
    t0 = time.perf_counter()
    pipeline.frame_tracker.update(gray)
    t1 = time.perf_counter()
    mi = pipeline.frame_tracker.last_match_info
    if fi > 0:
        print(f'  Frame {fi}: {t1-t0:.4f}s method={mi.get("method","?")} first_pts={mi.get("first_pts_count",0)}')
