import sys; sys.path.insert(0, '.')
import numpy as np, cv2, math
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
SPEED = 9.5
scene = SyntheticScene(render_proj, geom, speeds=[SPEED]*8)

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
pipeline.running = True
max_frames = int(geom.length / SPEED * fps) + 200
test_pt = np.array([[[960.0, 540.0]]], dtype=np.float64)

for fi in range(max_frames):
    rvec, tvec = perturb_jitter(fi, fps, r0, t0)
    render_proj.set_extrinsics(rvec, tvec)
    t = fi / fps
    athletes = scene.update(t)
    canvas = scene.render_background(athletes)
    dets = scene.get_detections(athletes)
    pipeline.process_frame(canvas, timestamp=t, external_detections=dets)
    if pipeline.timer.race_finished:
        break
    if fi % 200 == 0 and fi > 0:
        H = pipeline.frame_tracker.H_calib_current
        disp = cv2.perspectiveTransform(test_pt, H) - test_pt
        mi = pipeline.frame_tracker.last_match_info
        r_diff = np.linalg.norm(pipeline.projector.rvec - rvec)
        t_diff = np.linalg.norm(pipeline.projector.tvec - tvec)
        print(f'  frame {fi}: method={mi.get("method","?")}, H_disp={np.linalg.norm(disp[0,0]):.3f}px')
        print(f'          pose err: rvec={r_diff:.6f}, tvec={t_diff:.3f}, first_pts={mi.get("first_pts_count",0)}')

n_fin = len(pipeline.standings.finish_times)
print(f'Finished: {n_fin}/8, timer: {pipeline.timer.race_finished}, frames: {fi}')
dms = [float(a.d_m) for a in pipeline.assigner.athletes.values()]
print(f'Final d_ms: {dms}')
