import sys; sys.path.insert(0, '.')
import numpy as np, cv2, math
from calibration.coords import TrackGeometry, WorldCoord, ImageCoord
from calibration.projector import Projector
from pipeline.main_pipeline import TrackARPipeline
from tests.synthetic_scene import SyntheticScene
from tests.self_test import perturb_zoom

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
pipeline.running = True

render_proj = Projector(K, np.zeros((4,1)))
render_proj.set_extrinsics(r0.copy(), t0.copy())
scene = SyntheticScene(render_proj, geom, speeds=[9.5]*8)
fps = 60.0

dm_vals = []
corrected_count = 0
for fi in range(850):
    rvec, tvec = perturb_zoom(fi, fps, r0, t0)
    render_proj.set_extrinsics(rvec, tvec)
    t = fi / fps
    athletes = scene.update(t)
    canvas = scene.render_background(athletes)
    dets = scene.get_detections(athletes)
    pipeline.process_frame(canvas, timestamp=t, external_detections=dets)
    if 1 in pipeline.assigner.athletes:
        dm_vals.append(pipeline.assigner.athletes[1].d_m)
    mi = pipeline.frame_tracker.last_match_info
    if mi.get('drift_corrected'):
        corrected_count += 1
        if corrected_count <= 3:
            print(f'  fi={fi}: drift={mi.get("drift",0):.2f}, d_m_1={pipeline.assigner.athletes[1].d_m:.3f}')
    if fi % 100 == 0:
        if 1 in pipeline.assigner.athletes:
            print(f'  fi={fi}: d_m_1={pipeline.assigner.athletes[1].d_m:.3f}')
    if pipeline.timer.race_finished:
        print(f'Finish at frame {fi}')
        break

if not pipeline.timer.race_finished:
    print(f'Timer never stopped at frame {fi}')
    print(f'Final d_m lane 1: {dm_vals[-1]:.3f}')
    print(f'Drift corrections: {corrected_count}')
    print(f'First pts: {len(pipeline.frame_tracker._first_pts) if pipeline.frame_tracker._first_pts is not None else None}')
