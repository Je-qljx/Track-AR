import sys; sys.path.insert(0, '.')
import numpy as np, cv2, math, time
from calibration.coords import TrackGeometry, WorldCoord, ImageCoord
from calibration.projector import Projector
from tests.synthetic_scene import SyntheticScene

K = np.array([[700,0,960],[0,700,540],[0,0,1]], dtype=np.float64)
r0 = np.array([[-1.9865],[-0.7462],[-0.4312]], dtype=np.float64)
t0 = np.array([[-40.827],[-14.574],[36.127]], dtype=np.float64)
geom = TrackGeometry(track_type='100m')
proj = Projector(K, np.zeros((4,1)))
proj.set_extrinsics(r0.copy(), t0.copy())
scene = SyntheticScene(proj, geom, speeds=[9.5]*8)

fps = 60.0
t0 = time.time()
for fi in range(100):
    t = fi / fps
    athletes = scene.update(t)
    canvas = scene.render_background(athletes)
    dets = scene.get_detections(athletes)
t1 = time.time()
print(f'Render only: {t1-t0:.3f}s = {(t1-t0)/100*1000:.1f}ms/frame')

# Compare with real pipeline
from pipeline.main_pipeline import TrackARPipeline
pipeline = TrackARPipeline(camera_matrix=K, geometry=geom)
pts = geom.calibration_world_points()
wa = np.array([w.as_array for w in pts], dtype=np.float64)
pj, _ = cv2.projectPoints(wa, r0, t0, K, np.zeros((4,1)))
ip = [ImageCoord(float(p[0,0]), float(p[0,1])) for p in pj]
pipeline.calibrate_from_points(pts, ip)
track_pts = list(pts) + geom.build_tracking_grid(step=10.0)
pipeline.projector.set_calibration_world_pts(track_pts)
pipeline.running = True

t0 = time.time()
for fi in range(100):
    t = fi / fps
    athletes = scene.update(t)
    canvas = scene.render_background(athletes)
    dets = scene.get_detections(athletes)
    pipeline.process_frame(canvas, timestamp=t, external_detections=dets)
t1 = time.time()
print(f'Pipeline: {t1-t0:.3f}s = {(t1-t0)/100*1000:.1f}ms/frame')
