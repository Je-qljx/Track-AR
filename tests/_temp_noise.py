import sys; sys.path.insert(0, '.')
import numpy as np, cv2
from calibration.coords import TrackGeometry, WorldCoord, ImageCoord
from calibration.projector import Projector
from pipeline.main_pipeline import TrackARPipeline
from tests.synthetic_scene import SyntheticScene

K = np.array([[700,0,960],[0,700,540],[0,0,1]], dtype=np.float64)
r0 = np.array([[-1.9865],[-0.7462],[-0.4312]], dtype=np.float64)
t0 = np.array([[-40.827],[-14.574],[36.127]], dtype=np.float64)
geom = TrackGeometry(track_type='100m')
pipeline = TrackARPipeline(camera_matrix=K, geometry=geom)

# Apply 3px noise to calibration
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

# Check calibration error
err = pipeline.calibrator.get_projection_error(pts, image_pts)
print(f'Calibration reprojection error: {err:.3f}px')

# Check finish distance projection
for lane in range(1, 9):
    fd = geom.finish_distance(lane)
    wc = geom.world_coord(lane, fd)
    ic = pipeline.projector.project(wc)
    print(f'  Lane {lane}: finish at ({ic.u:.1f}, {ic.v:.1f})')

render_proj = Projector(K, np.zeros((4,1)))
render_proj.set_extrinsics(r0.copy(), t0.copy())
scene = SyntheticScene(render_proj, geom, speeds=[9.5]*8)
fps = 60.0
pipeline.running = True
max_frames = int(100 / 9.5 * 60) + 200

drop_count = 0
missed_by_lane = {lane: 0 for lane in range(1,9)}

for fi in range(max_frames):
    t = fi / fps
    athletes = scene.update(t)
    canvas = scene.render_background(athletes)
    dets = scene.get_detections(athletes)
    pipeline.process_frame(canvas, timestamp=t, external_detections=dets)
    active = [a for a in pipeline.assigner.athletes.values() if a.tracking_confidence > 0]
    if fi >= 10 and len(active) < 8:
        drop_count += 1
        missing = set(range(1,9)) - set(a.lane for a in active)
        for l in missing:
            missed_by_lane[l] = missed_by_lane.get(l, 0) + 1
    if pipeline.timer.race_finished:
        break

print(f'Drops: {drop_count} in {fi+1} frames')
print(f'Missed by lane: {missed_by_lane}')
n_fin = len(pipeline.standings.finish_times)
print(f'Finished: {n_fin}/8')
dms = [float(a.d_m) for a in pipeline.assigner.athletes.values()]
print(f'Final d_ms: {dms}')
