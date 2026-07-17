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

# Force H to identity (skip frame tracker)
pipeline.projector._tracker.H_calib_current = np.eye(3, dtype=np.float64)

def perturb_dolly(f, fps, r0, t0):
    t = t0.copy()
    t[0, 0] += 0.5 * math.sin(2*math.pi*f/(fps*15.0))
    return r0.copy(), t

render_proj = Projector(K, np.zeros((4,1)))
render_proj.set_extrinsics(r0.copy(), t0.copy())
scene = SyntheticScene(render_proj, geom, speeds=[9.5]*8)
fps = 60.0
pipeline.running = True

dm_vals = []
for fi in range(700):
    rvec, tvec = perturb_dolly(fi, fps, r0, t0)
    render_proj.set_extrinsics(rvec, tvec)
    t = fi / fps
    athletes = scene.update(t)
    canvas = scene.render_background(athletes)
    dets = scene.get_detections(athletes)
    pipeline.process_frame(canvas, timestamp=t, external_detections=dets)
    if 1 in pipeline.assigner.athletes:
        dm_vals.append(pipeline.assigner.athletes[1].d_m)
    if pipeline.timer.race_finished:
        print(f'Finish frame: {fi}')
        break

print(f'Without tracker (H=identity): finish at frame {fi}')
for lane in range(1, 9):
    if lane in pipeline.assigner.athletes:
        print(f'  Lane {lane}: dm={pipeline.assigner.athletes[lane].d_m:.3f}')
