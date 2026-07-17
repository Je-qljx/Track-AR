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
pts = geom.calibration_world_points()
wa = np.array([w.as_array for w in pts], dtype=np.float64)
pj, _ = cv2.projectPoints(wa, r0, t0, K, np.zeros((4,1)))
ip = [ImageCoord(float(p[0,0]), float(p[0,1])) for p in pj]
pipeline.calibrate_from_points(pts, ip)
track_pts = list(pts) + [WorldCoord(dm, geom.lane_center_y(l), 0.0) for l in range(1,9) for dm in range(0,101,10)]
pipeline.projector.set_calibration_world_pts(track_pts)

render_proj = Projector(K, np.zeros((4,1)))
render_proj.set_extrinsics(r0.copy(), t0.copy())
scene = SyntheticScene(render_proj, geom, speeds=[9.5]*8)
fps = 60.0
pipeline.running = True
max_frames = int(100 / 9.5 * 60) + 200

# Patch to trace lane 5 matching
orig_match = pipeline.assigner._match_existing_athlete
missed_count = 0

def traced_match(athlete, detections, frame_dt, dets_used=None):
    global missed_count
    result = orig_match(athlete, detections, frame_dt, dets_used)
    if athlete.lane == 5:
        if result is None:
            missed_count += 1
            # Log first 3 misses with details
            if missed_count <= 3:
                print(f'  FRAME {pipeline.assigner._frame_count}: Lane 5 missed! '
                      f'd_m={athlete.d_m:.1f}, coast={athlete.coast_count}, '
                      f'conf={athlete.tracking_confidence:.2f}, n_dets={len(detections)}')
                # Show what detections were available near lane 5
                pu, pv = pipeline.assigner._predict_pixel_current(athlete)
                for di, d in enumerate(detections):
                    du, dv = d.bottom_center
                    dp = np.sqrt((du-pu)**2 + (dv-pv)**2)
                    if dp < 200:
                        likely = pipeline.assigner._is_likely_athlete(d)
                        print(f'    det {di}: pos=({du:.0f},{dv:.0f}), dist={dp:.0f}px, '
                              f'conf={d.confidence:.2f}, bbox={d.bbox}, likely={likely}')
    return result

pipeline.assigner._match_existing_athlete = traced_match

for fi in range(max_frames):
    t = fi / fps
    athletes = scene.update(t)
    canvas = scene.render_background(athletes)
    dets = scene.get_detections(athletes)
    fp = scene.generate_spurious_detections(count=30, seed=fi)
    all_dets = dets + fp
    pipeline.process_frame(canvas, timestamp=t, external_detections=all_dets)
    if pipeline.timer.race_finished:
        break

print(f'Total lane 5 misses: {missed_count}')
print(f'Race finished: {pipeline.timer.race_finished}')
dms = [float(a.d_m) for a in pipeline.assigner.athletes.values()]
print(f'Final d_ms: {dms}')
