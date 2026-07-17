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
rng = np.random.RandomState(42)
image_pts = []
for p in pj:
    u = float(p[0, 0]) + rng.randn() * 3.0
    v = float(p[0, 1]) + rng.randn() * 3.0
    image_pts.append(ImageCoord(u, v))
pipeline.calibrate_from_points(pts, image_pts)
track_pts = list(pts) + [WorldCoord(dm, geom.lane_center_y(l), 0.0) for l in range(1,9) for dm in range(0,101,10)]
pipeline.projector.set_calibration_world_pts(track_pts)

render_proj = Projector(K, np.zeros((4,1)))
render_proj.set_extrinsics(r0.copy(), t0.copy())
scene = SyntheticScene(render_proj, geom, speeds=[9.5]*8)
fps = 60.0
pipeline.running = True
max_frames = int(100 / 9.5 * 60) + 200

for fi in range(min(max_frames, 50)):
    t = fi / fps
    athletes = scene.update(t)
    canvas = scene.render_background(athletes)
    dets = scene.get_detections(athletes)
    pipeline.process_frame(canvas, timestamp=t, external_detections=dets)

# Check distance_traveled for lane 1-8
for lane in range(1, 9):
    if lane in pipeline.assigner.athletes:
        a = pipeline.assigner.athletes[lane]
        print(f'  Lane {lane}: d_m={a.d_m:.2f}, dist_traveled={a.distance_traveled:.4f}, n_frames={a.frames_tracked}')
    else:
        print(f'  Lane {lane}: NOT IN ASSIGNER')
