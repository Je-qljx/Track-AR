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

def run_noise3px():
    pipeline = TrackARPipeline(camera_matrix=K, geometry=geom)
    pts = geom.calibration_world_points()
    wa = np.array([w.as_array for w in pts], dtype=np.float64)
    pj, _ = cv2.projectPoints(wa, r0, t0, K, np.zeros((4,1)))
    rng = np.random.RandomState(42)
    image_pts = [ImageCoord(float(p[0,0])+rng.randn()*3.0, float(p[0,1])+rng.randn()*3.0) for p in pj]
    pipeline.calibrate_from_points(pts, image_pts)
    track_pts = list(pts) + pipeline.geometry.build_tracking_grid(step=10.0)
    pipeline.projector.set_calibration_world_pts(track_pts)
    render_proj = Projector(K, np.zeros((4,1)))
    render_proj.set_extrinsics(r0.copy(), t0.copy())
    scene = SyntheticScene(render_proj, geom, speeds=[9.5]*8)
    
    pipeline.running = True
    fps = 60.0
    for fi in range(int(100/9.5*60)+200):
        t = fi / fps
        athletes = scene.update(t)
        canvas = scene.render_background(athletes)
        dets = scene.get_detections(athletes)
        pipeline.process_frame(canvas, timestamp=t, external_detections=dets)
        if pipeline.timer.race_finished:
            print(f'  noise3px: finish frame {fi}')
            break
    print(f'  Finished: {len(pipeline.standings.finish_times)}/8')

def run_pan():
    pipeline = TrackARPipeline(camera_matrix=K, geometry=geom)
    pts = geom.calibration_world_points()
    wa = np.array([w.as_array for w in pts], dtype=np.float64)
    pj, _ = cv2.projectPoints(wa, r0, t0, K, np.zeros((4,1)))
    ip = [ImageCoord(float(p[0,0]), float(p[0,1])) for p in pj]
    pipeline.calibrate_from_points(pts, ip)
    track_pts = list(pts) + pipeline.geometry.build_tracking_grid(step=10.0)
    pipeline.projector.set_calibration_world_pts(track_pts)
    render_proj = Projector(K, np.zeros((4,1)))
    render_proj.set_extrinsics(r0.copy(), t0.copy())
    scene = SyntheticScene(render_proj, geom, speeds=[9.5]*8)
    
    def pan(f, fps, r, t):
        rv = r.copy()
        rv[1,0] += 0.0003 * math.sin(2*math.pi*f/(fps*10))
        return rv, t.copy()
    
    pipeline.running = True
    fps = 60.0
    for fi in range(int(100/9.5*60)+200):
        rvec, tvec = pan(fi, fps, r0, t0)
        render_proj.set_extrinsics(rvec, tvec)
        t = fi / fps
        athletes = scene.update(t)
        canvas = scene.render_background(athletes)
        dets = scene.get_detections(athletes)
        pipeline.process_frame(canvas, timestamp=t, external_detections=dets)
        if pipeline.timer.race_finished:
            print(f'  pan: finish frame {fi}')
            break
    print(f'  Finished: {len(pipeline.standings.finish_times)}/8')

run_noise3px()
print()
run_pan()
