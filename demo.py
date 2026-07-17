"""
TrackAR — Track & Field Real-time AR Overlay System
"""
import sys
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibration.coords import TrackGeometry, ImageCoord
from pipeline.main_pipeline import TrackARPipeline
from ui.control_panel import ControlPanel
from tests.synthetic_scene import SyntheticScene


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=["100m", "400m"], default="100m")
    args = parser.parse_args()

    is_400m = args.track == "400m"
    print("=" * 60)
    print(f"  TrackAR — {args.track} Track & Field Real-time AR Overlay")
    print("  Demo Mode")
    print("=" * 60)

    if is_400m:
        K = np.array([[700, 0, 960], [0, 700, 540], [0, 0, 1]], dtype=np.float64)
        geom = TrackGeometry(track_type="400m")
        pipeline = TrackARPipeline(camera_matrix=K, geometry=geom)
        rvec = np.array([[0.6], [0.0], [0.0]], dtype=np.float64)
        tvec = np.array([[-10], [5], [90]], dtype=np.float64)
        speeds = [9.3, 9.5, 9.7, 9.9, 9.1, 8.8, 9.4, 10.1]
    else:
        K = np.array([[700, 0, 960], [0, 700, 540], [0, 0, 1]], dtype=np.float64)
        geom = TrackGeometry(track_type="100m")
        pipeline = TrackARPipeline(camera_matrix=K, geometry=geom)
        rvec = np.array([[-1.9865], [-0.7462], [-0.4312]], dtype=np.float64)
        tvec = np.array([[-40.827], [-14.574], [36.127]], dtype=np.float64)
        speeds = [9.5, 9.8, 10.2, 9.0, 8.5, 9.3, 8.8, 10.5]

    w_pts = geom.calibration_world_points()
    w_arr = np.array([w.as_array for w in w_pts], dtype=np.float64)
    proj, _ = cv2.projectPoints(w_arr, rvec, tvec, K, np.zeros((4, 1)))
    i_pts = [ImageCoord(float(p[0, 0]), float(p[0, 1])) for p in proj]
    pipeline.calibrate_from_points(w_pts, i_pts)
    # Use a separate projector for scene rendering so follow-mode camera
    # modifications don't affect tracking
    render_proj = Projector(K)
    render_proj.set_extrinsics(rvec.copy(), tvec.copy())
    scene = SyntheticScene(render_proj, geom, speeds=speeds)
    pipeline.dynamic_camera._render_proj = render_proj
    for lane in range(1, 9):
        pipeline.set_athlete_name(lane, f"Athlete {lane}")
    print("[OK] Pipeline initialized and calibrated")
    print(f"[OK] Synthetic scene ready (8 athletes, 60fps, {args.track})")
    print("\nControls:")
    print("  [p] Pause    [b] Toggle bboxes    [o] Toggle overlay    [f] Follow mode    [r] Reset    [q] Quit")
    print("-" * 60)
    control = ControlPanel(pipeline)
    control.create_trackbars()
    pipeline.running = True
    frame_idx = 0
    max_frames = int(geom.race_length() / max(speeds) * 60 * 1.2)

    def on_reset():
        nonlocal frame_idx
        frame_idx = 0

    control.on_reset = on_reset
    while pipeline.running:
        if control.state.paused:
            key = cv2.waitKey(100) & 0xFF
            if key > 0:
                control.handle_key(key)
            continue
            t = frame_idx / 60.0
            athletes = scene.update(t)
            # Sync render projector with current tracking pose + follow adjustments
            render_proj.set_extrinsics(pipeline.projector.rvec.copy(), pipeline.projector.tvec.copy())
            if pipeline.dynamic_camera.follow_mode and hasattr(pipeline.dynamic_camera, '_render_proj'):
                dc = pipeline.dynamic_camera
                look_x = dc.compute_look_x(pipeline._last_positions if hasattr(pipeline, '_last_positions') else [], pipeline.geometry.length)
                if dc.prev_look_x is None:
                    dc.prev_look_x = look_x
                dc.prev_look_x += (look_x - dc.prev_look_x) * dc.SMOOTH_ALPHA
                geom = pipeline.geometry
                render_proj.look_at(WorldCoord(dc.prev_look_x, geom.lane_center_y(4.5), 0.0))
            canvas = scene.render(athletes)
            detections = scene.get_detections(athletes)
            output = pipeline.process_frame(canvas, timestamp=t, external_detections=detections)
            if control.state.overlay_enabled:
                control.draw_controls(output)
            cv2.imshow("TrackAR Demo", output)
            key = cv2.waitKey(1) & 0xFF
            if key > 0:
                if not control.handle_key(key):
                    pipeline.running = False
            frame_idx += 1
        if frame_idx > max_frames:
            frame_idx = 0
    cv2.destroyAllWindows()
    print("\nDemo finished.")


if __name__ == "__main__":
    main()
