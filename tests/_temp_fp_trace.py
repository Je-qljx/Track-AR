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

# Instrument the assigner to trace lane 5
orig_pf = pipeline.assigner.process_frame

def traced_pf(detections, frame_dt=1.0/60.0):
    # Check if lane 5's real detection is present
    lane5_athletes = [a for a in pipeline.assigner.athletes.values() if a.lane == 5]
    lane5_ath = lane5_athletes[0] if lane5_athletes else None
    
    if lane5_ath:
        # Get predicted position for lane 5
        pu, pv = pipeline.assigner._predict_pixel_current(lane5_ath)
        
        # Find real detection for lane 5 among detections
        real_det = None
        for d in detections:
            du, dv = d.bottom_center
            if abs(du - pu) < 5 and abs(dv - pv) < 5 and d.confidence == 0.95:
                real_det = d
                break
        
        if real_det:
            # Check if real detection passes filters
            likely = pipeline.assigner._is_likely_athlete(real_det)
            u_raw, v_raw = real_det.bottom_center
            u_foot, v_foot = pipeline.assigner._current_to_calib(u_raw, v_raw)
            in_track = pipeline.assigner._is_in_track_region(u_foot, v_foot)
            
            # Check if NMS suppresses it
            filtered = [d for d in detections if pipeline.assigner._is_likely_athlete(d)]
            if not 100:
                filtered = [d for d in filtered if pipeline.assigner._is_in_track_region(*pipeline.assigner._current_to_calib(*d.bottom_center))]
            
            # NMS
            if len(filtered) > 1:
                filtered.sort(key=lambda d: d.confidence, reverse=True)
                keep = [True] * len(filtered)
                for i in range(len(filtered)):
                    if not keep[i]: continue
                    bi = filtered[i].bbox
                    ai_area = max((bi[2]-bi[0]),1) * max((bi[3]-bi[1]),1)
                    for j in range(i+1, len(filtered)):
                        if not keep[j]: continue
                        bj = filtered[j].bbox
                        x1 = max(bi[0], bj[0])
                        y1 = max(bi[1], bj[1])
                        x2 = min(bi[2], bj[2])
                        y2 = min(bi[3], bj[3])
                        if x2 < x1 or y2 < y1: continue
                        inter = (x2-x1)*(y2-y1)
                        aj_area = max((bj[2]-bj[0]),1) * max((bj[3]-bj[1]),1)
                        iou = inter / (ai_area + aj_area - inter + 1e-6)
                        if iou > 0.85:
                            keep[j] = False
                nms_filtered = [d for d, k in zip(filtered, keep) if k]
            else:
                nms_filtered = filtered
            
            real_in_nms = any(d is real_det for d in nms_filtered)
            
            if not real_in_nms:
                print(f'  FRAME {pipeline.assigner._frame_count}: lane5 real detection SUPPRESSED! likely={likely} in_track={in_track}')
                
    return orig_pf(detections, frame_dt)

pipeline.assigner.process_frame = traced_pf

# Run test
for fi in range(60):
    t = fi / fps
    athletes = scene.update(t)
    canvas = scene.render_background(athletes)
    dets = scene.get_detections(athletes)
    fp = scene.generate_spurious_detections(count=30, seed=fi)
    all_dets = dets + fp
    pipeline.process_frame(canvas, timestamp=t, external_detections=all_dets)
    if pipeline.timer.race_finished:
        break

print('Done')
