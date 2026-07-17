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

# Trace lane 5's existence and confidence
lanes_removed = 0
for fi in range(max_frames):
    t = fi / fps
    athletes = scene.update(t)
    canvas = scene.render_background(athletes)
    dets = scene.get_detections(athletes)
    fp = scene.generate_spurious_detections(count=30, seed=fi)
    all_dets = dets + fp
    pipeline.process_frame(canvas, timestamp=t, external_detections=all_dets)
    
    # Check lane 5 state
    has5 = 5 in pipeline.assigner.athletes
    if has5:
        conf5 = pipeline.assigner.athletes[5].tracking_confidence
        fm5 = pipeline.assigner.athletes[5].frames_missed
        if fi > 10 and conf5 == 0:
            print(f'  FRAME {fi}: lane5 conf=0, frames_missed={fm5}, d_m={pipeline.assigner.athletes[5].d_m:.1f}')
        elif fi > 10 and fm5 > 500:
            print(f'  FRAME {fi}: lane5 frames_missed={fm5}, conf={conf5:.2f}')
    else:
        if fi > 10:
            print(f'  FRAME {fi}: lane5 NOT in athletes!')
            lanes_removed += 1
    
    if pipeline.timer.race_finished:
        break

print(f'Lane 5 was removed {lanes_removed} times')
print(f'Final: race_finished={pipeline.timer.race_finished}')
dms = [float(a.d_m) for a in pipeline.assigner.athletes.values()]
print(f'Final d_ms: {dms}')
