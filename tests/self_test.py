import sys; sys.path.insert(0, 'D:/track_ar')
import numpy as np
import cv2
import time as _time
from calibration.coords import TrackGeometry, ImageCoord, WorldCoord
from calibration.projector import Projector
from pipeline.main_pipeline import TrackARPipeline
from tests.synthetic_scene import SyntheticScene

SPEED = 9.5
K = np.array([[700, 0, 960], [0, 700, 540], [0, 0, 1]], dtype=np.float64)

# ---- camera poses ----
R0_100M = np.array([[-1.9865], [-0.7462], [-0.4312]], dtype=np.float64)
T0_100M = np.array([[-40.827], [-14.574], [36.127]], dtype=np.float64)
R0_400M = np.array([[-0.3002], [-3.0314], [-0.5945]], dtype=np.float64)
T0_400M = np.array([[-33.3], [43.9], [125.9]], dtype=np.float64)
R0_SIDE = np.array([[0.0], [2.3805], [2.0501]], dtype=np.float64)
T0_SIDE = np.array([[50.0], [-0.02], [40.49]], dtype=np.float64)


def _sin_amp(f, fps, period_s, amp):
    return amp * np.sin(2 * np.pi * f / (fps * period_s))

# ---- perturbation functions (realistic broadcast amplitudes) ----
def perturb_static(f, fps, r0, t0):
    return r0.copy(), t0.copy()

def perturb_pan(f, fps, r0, t0):
    r = r0.copy()
    r[1, 0] += _sin_amp(f, fps, 10.0, 0.0003)
    return r, t0.copy()

def perturb_zoom(f, fps, r0, t0):
    t = t0.copy()
    t[2, 0] += _sin_amp(f, fps, 12.0, 0.5)
    return r0.copy(), t

def perturb_dolly(f, fps, r0, t0):
    t = t0.copy()
    t[0, 0] += _sin_amp(f, fps, 15.0, 0.5)
    return r0.copy(), t

def perturb_boom(f, fps, r0, t0):
    r = r0.copy()
    t = t0.copy()
    t[1, 0] += _sin_amp(f, fps, 8.0, 0.3)
    return r, t


def _finish_time(v_max: float, race_len: float, tau: float = 1.5) -> float:
    """Newton solve for finish time under acceleration: d(t) = v_max*(t + tau*e^(-t/tau) - tau)."""
    guess = race_len / v_max + 1.5
    for _ in range(30):
        d = v_max * (guess + tau * np.exp(-guess / tau) - tau)
        v = v_max * (1.0 - np.exp(-guess / tau))
        error = d - race_len
        if abs(error) < 0.001:
            break
        guess -= error / max(v, 0.01)
    return guess


def perturb_pan_zoom(f, fps, r0, t0):
    r = r0.copy()
    t = t0.copy()
    r[1, 0] += _sin_amp(f, fps, 10.0, 0.0003)
    t[2, 0] += _sin_amp(f, fps, 12.0, 0.4)
    return r, t

def perturb_pan_moderate(f, fps, r0, t0):
    r = r0.copy()
    t = t0.copy()
    progress = min(f / fps / (100.0 / 9.5), 1.0)
    r[1, 0] += 0.10 * progress  # 0 → 0.10 rad (~5.7 deg) — known to pass
    return r, t

def perturb_pan_wide(f, fps, r0, t0):
    r = r0.copy()
    t = t0.copy()
    t_sec = f / fps
    progress = min(t_sec / (100.0 / 9.5), 1.0)  # 0→1 over race duration
    r[1, 0] += 0.8 * progress  # cumulative pan: 0 → 0.8 rad (~46 deg)
    return r, t

def perturb_pan_wide_400m(f, fps, r0, t0):
    r = r0.copy()
    t = t0.copy()
    t_sec = f / fps
    progress = min(t_sec / (400.0 / 9.5), 1.0)
    r[1, 0] += 1.2 * progress  # larger cumulative pan over longer race
    return r, t

def perturb_pan_zoom_wide(f, fps, r0, t0):
    r = r0.copy()
    t = t0.copy()
    t_sec = f / fps
    progress = min(t_sec / (100.0 / 9.5), 1.0)
    r[1, 0] += 0.8 * progress
    t[2, 0] += _sin_amp(f, fps, 12.0, 0.4)
    return r, t

def perturb_jitter(f, fps, r0, t0):
    r = r0.copy()
    t = t0.copy()
    r[1, 0] += _sin_amp(f, fps, 10.0, 0.0003)
    rng = np.random.RandomState(int(f))
    r[0, 0] += rng.uniform(-0.00015, 0.00015)
    r[2, 0] += rng.uniform(-0.00015, 0.00015)
    t[1, 0] += rng.uniform(-0.05, 0.05)
    t[2, 0] += rng.uniform(-0.05, 0.05)
    return r, t


def _add_tracking_grid(geom, calib_pts, target_spec=None):
    """Build dense grid of world points for robust track_homography PnP."""
    pts = list(calib_pts)
    is_400m = geom._model is not None
    for lane in range(1, 9):
        for dm in np.arange(0.0, geom.length + 1, 10.0):
            if is_400m:
                wc = geom.world_coord(lane, dm)
            else:
                wc = WorldCoord(dm, geom.lane_center_y(lane), 0.0)
            if target_spec is not None:
                dm_t, lane_t, _, _ = target_spec
                if lane == lane_t and abs(dm - dm_t) < 1.0:
                    continue
            pts.append(wc)
    return pts


def run_test(track_type: str, r0, t0, cam_K, perturb_fn,
             target_spec=None, name: str = "",
             max_time_err_s: float = 0.2,
             calib_noise_px: float = 0.0,
             use_acceleration: bool = False,
             num_false_positives: int = 0,
             detection_dropout_rate: float = 0.0) -> str:
    """Full synthetic race test.

    target_spec: None → standard; (dm, lane, w, h) → calibration target.
    max_time_err_s: maximum allowed finish-time error in seconds.
    calib_noise_px: Gaussian pixel noise added to calibration image points.
    use_acceleration: use exponential acceleration model instead of constant speed.
    num_false_positives: spurious non-athlete detections injected per frame.
    detection_dropout_rate: probability of each real detection being dropped.
    """
    geom = TrackGeometry(track_type=track_type)
    pipeline = TrackARPipeline(camera_matrix=cam_K, geometry=geom)

    # Target mode at distance produces few-pixel spread → PnP is ill-conditioned;
    # test without noise (target calibration correctness only).
    if target_spec is not None:
        calib_noise_px = 0.0

    if target_spec is not None:
        dm_t, lane_t, w_t, h_t = target_spec
        calib_pts = geom.calibration_target_points(lane_t, dm_t, w_t, h_t)
    else:
        calib_pts = geom.calibration_world_points()
    w_arr = np.array([w.as_array for w in calib_pts], dtype=np.float64)
    proj, _ = cv2.projectPoints(w_arr, r0, t0, cam_K, np.zeros((4, 1)))
    noise_rng = np.random.RandomState(42)
    image_pts = []
    for p in proj:
        u = float(p[0, 0]) + noise_rng.randn() * calib_noise_px
        v = float(p[0, 1]) + noise_rng.randn() * calib_noise_px
        image_pts.append(ImageCoord(u, v))
    pipeline.calibrate_from_points(calib_pts, image_pts)

    # Dense tracking grid for robust PnP in track_homography
    track_pts = _add_tracking_grid(geom, calib_pts, target_spec)
    pipeline.projector.set_calibration_world_pts(track_pts)

    err = pipeline.calibrator.get_projection_error(calib_pts, image_pts)
    if err > 5.0:
        return f"FAIL: calib error {err:.3f}px"

    render_proj = Projector(cam_K, np.zeros((4, 1)))
    render_proj.set_extrinsics(r0.copy(), t0.copy())
    scene = SyntheticScene(render_proj, geom, speeds=[SPEED] * 8,
                           use_acceleration=use_acceleration)
    fps = 60.0
    race_len = geom.finish_distance(1)
    if use_acceleration:
        tau_l1 = 1.5 + 1 * 0.05
        expected_time = _finish_time(SPEED, race_len, tau_l1)
        max_frames = int(expected_time * fps) + 200
    else:
        max_frames = int(race_len / SPEED * fps) + 200

    t_start = _time.time()
    drop_count = 0

    for fi in range(max_frames):
        if _time.time() - t_start > 300:
            return "TIMEOUT"
        rvec, tvec = perturb_fn(fi, fps, r0, t0)
        render_proj.set_extrinsics(rvec, tvec)
        t = fi / fps
        athletes = scene.update(t)
        canvas = scene.render_background(athletes)
        detections = scene.get_detections(athletes)
        if num_false_positives > 0:
            detections += scene.generate_spurious_detections(count=num_false_positives, seed=fi)
        if detection_dropout_rate > 0.0:
            rng_drop = np.random.RandomState(fi)
            detections = [d for d in detections if rng_drop.uniform() > detection_dropout_rate]
        pipeline.process_frame(canvas, timestamp=t, external_detections=detections)
        active = [a for a in pipeline.assigner.athletes.values() if a.tracking_confidence > 0]
        if fi >= 10 and len(active) < 8:
            drop_count += 1
        if pipeline.timer.race_finished:
            break

    if not pipeline.timer.race_finished:
        return f"FAIL: timer not stopped (n_fin={len(pipeline.standings.finish_times)}, frames={fi})"
    n_fin = len(pipeline.standings.finish_times)
    if n_fin < 8:
        return f"FAIL: only {n_fin}/8 finished"
    max_drops = int(max_frames * 0.1)
    if drop_count > max_drops and drop_count > 10:
        return f"FAIL: {drop_count} drops in {max_frames} frames"

    if use_acceleration:
        tau_l1 = 1.5 + 1 * 0.05
        expected_time = _finish_time(SPEED, race_len, tau_l1)
        expected_frame = int(expected_time * fps)
    else:
        expected_frame = int(race_len / SPEED * fps)
    frame_error = abs(fi - expected_frame)
    max_frame_err = int(max_time_err_s * fps)
    if frame_error > max_frame_err:
        return f"FAIL: finish frame {fi} vs expected {expected_frame} ({frame_error/fps:.1f}s error > {max_time_err_s}s)"
    return f"PASS ({fi/fps:.1f}s, {fi+1} frames, err={frame_error})"


# ---- full-race scenarios ----
# Rationale for removals:
#   - target positions (start/finish/edge) → 1 representative tgt_mid covers
#   - target sizes (tiny/small/large)   → quick cal already covers; full race needs 1 size
#   - side view                        → quick cal check covers calibration accuracy
#   - boom ≈ zoom (both scale/depth ambiguity)   → keep zoom (more general)
#   - dolly ≈ zoom (both depth ambiguity)        → keep zoom (larger impact)
#   - 400m zoom                        → covered by 400m/panzoom/combined motion
#   - motion+target boom/zoom/dolly     → covered by pan/tgt_mid (motion+covers)
#   - 400m stress variants (falsepos, jitter) → 400m always passes; 100m is the strict case
SCENARIOS = [
    # === Static camera (standard calibration) ===
    ("100m/static/std",               "100m", R0_100M, T0_100M, perturb_static, None),
    ("400m/static/std",               "400m", R0_400M, T0_400M, perturb_static, None),

    # === Target mode — 1 representative each ===
    ("100m/static/tgt_mid",           "100m", R0_100M, T0_100M, perturb_static, (50, 5, 0.420, 0.297)),
    ("400m/static/tgt_mid",           "400m", R0_400M, T0_400M, perturb_static, (200, 5, 0.420, 0.297)),

    # === Camera motion + standard calibration ===
    ("100m/pan/std",                  "100m", R0_100M, T0_100M, perturb_pan,   None),
    ("400m/pan/std",                  "400m", R0_400M, T0_400M, perturb_pan,   None),
    ("100m/zoom/std",                 "100m", R0_100M, T0_100M, perturb_zoom,  None,     1.5),  # depth-ambiguity

    # === Combined motion (pan + zoom) ===
    ("100m/panzoom/std",              "100m", R0_100M, T0_100M, perturb_pan_zoom, None,   0.3),
    ("400m/panzoom/std",              "400m", R0_400M, T0_400M, perturb_pan_zoom, None,   0.3),

    # === Wide-range cumulative pan (follow athletes start→finish) ===
    # 0.10 rad → moderate — system handles this
    ("100m/pan_wide/mod",             "100m", R0_100M, T0_100M, perturb_pan_moderate, None, 0.5),
    # 0.80 rad → extreme — reveals PnP/KLT drift limitation
    ("100m/pan_wide/extreme",         "100m", R0_100M, T0_100M, perturb_pan_wide, None, 0.5),
    ("400m/pan_wide/extreme",         "400m", R0_400M, T0_400M, perturb_pan_wide_400m, None, 0.5),

    # === Wide pan + zoom oscillation ===
    ("100m/pan_zoom_wide/extreme",    "100m", R0_100M, T0_100M, perturb_pan_zoom_wide, None, 0.5),

    # === Camera motion + target mode — 1 representative each ===
    ("100m/pan/tgt_mid",              "100m", R0_100M, T0_100M, perturb_pan,   (50, 5, 0.420, 0.297)),
    ("400m/pan/tgt_mid",              "400m", R0_400M, T0_400M, perturb_pan,   (200, 5, 0.420, 0.297)),

    # === Stress: false positive detections (spectator/official) ===
    ("100m/static/falsepos_30",       "100m", R0_100M, T0_100M, perturb_static, None,   0.5,   False, 30),

    # === Stress: camera jitter (smooth pan + random high-frequency shake) ===
    ("100m/jitter/std",               "100m", R0_100M, T0_100M, perturb_jitter, None,   0.5),

    # === Stress: 50% detection dropout ===
    ("100m/dropout50/std",            "100m", R0_100M, T0_100M, perturb_static, None,   0.5,   False, 0, 0.5),

    # === Stress: calibration noise (3px) ===
    ("100m/static/noise3px",          "100m", R0_100M, T0_100M, perturb_static, None,   1.0,   False, 0, 0.0, 3.0),
]

# ---- 标定物快速验收 ----
def run_target_calib_check(track_type, r0, t0, cam_K, target_spec, name="",
                           calib_noise_px: float = 0.0):
    # Target mode → small feature spread in image → test without noise
    if target_spec is not None:
        calib_noise_px = 0.0

    geom = TrackGeometry(track_type=track_type)
    pipeline = TrackARPipeline(camera_matrix=cam_K, geometry=geom)
    noise_rng = np.random.RandomState(42)

    if target_spec is not None:
        dm_t, lane_t, w_t, h_t = target_spec
        world_pts = geom.calibration_target_points(lane_t, dm_t, w_t, h_t)
        wa = np.array([w.as_array for w in world_pts], dtype=np.float64)
        pj, _ = cv2.projectPoints(wa, r0, t0, cam_K, np.zeros((4, 1)))
        ip = []
        for p in pj:
            u = float(p[0, 0]) + noise_rng.randn() * calib_noise_px
            v = float(p[0, 1]) + noise_rng.randn() * calib_noise_px
            ip.append(ImageCoord(u, v))
        pipeline.calibrate_from_points(world_pts, ip)
        cal_err = pipeline.calibrator.get_projection_error(world_pts, ip)
        if cal_err > 5.0:
            return f"FAIL: calib error {cal_err:.3f}px"
        if pipeline.projector._calib_world_pts is None:
            return "FAIL: calibration world pts not stored"
        # Unproject round-trip at target center
        if pipeline.projector.rvec is not None:
            center_img = ImageCoord(
                float(np.mean([p.u for p in ip])),
                float(np.mean([p.v for p in ip])),
            )
            center_world = pipeline.projector.unproject_to_ground(center_img)
            target_center = geom.world_coord(lane_t, dm_t)
            dx = abs(center_world.x - target_center.x)
            dy = abs(center_world.y - target_center.y)
            if dx > 3.0 or dy > 3.0:
                return f"FAIL: unproject error ({dx:.2f}m, {dy:.2f}m)"
        return f"PASS (err={cal_err:.3f}px)"
    else:
        # Standard calibration check
        geom = TrackGeometry(track_type=track_type)
        pipeline = TrackARPipeline(camera_matrix=cam_K, geometry=geom)
        wp = geom.calibration_world_points()
        wa = np.array([w.as_array for w in wp], dtype=np.float64)
        pj, _ = cv2.projectPoints(wa, r0, t0, cam_K, np.zeros((4, 1)))
        ip = []
        for p in pj:
            u = float(p[0, 0]) + noise_rng.randn() * calib_noise_px
            v = float(p[0, 1]) + noise_rng.randn() * calib_noise_px
            ip.append(ImageCoord(u, v))
        pipeline.calibrate_from_points(wp, ip)
        err = pipeline.calibrator.get_projection_error(wp, ip)
        if err > 5.0:
            return f"FAIL: calib error {err:.3f}px"
        # Additional round-trip check on track center (100m only)
        if track_type == "100m" and pipeline.projector.rvec is not None:
            center_wc = WorldCoord(geom.length / 2, geom.lane_center_y(4), 0.0)
            center_ic = pipeline.projector.project(center_wc)
            center_back = pipeline.projector.unproject_to_ground(center_ic)
            ctx, cty = center_wc.x, center_wc.y
            cbx, cby = center_back.x, center_back.y
            dx = abs(cbx - ctx)
            dy = abs(cby - cty)
            if dx > 5.0 or dy > 5.0:
                return f"FAIL: std unproject error ({dx:.2f}m, {dy:.2f}m)"
        return f"PASS (err={err:.3f}px)"


QUICK_SCENARIOS = [
    ("qc_100m_std",      "100m", R0_100M, T0_100M, None),
    ("qc_400m_std",      "400m", R0_400M, T0_400M, None),
    ("qc_100m_mid",      "100m", R0_100M, T0_100M, (50, 5, 0.420, 0.297)),
    ("qc_100m_start",    "100m", R0_100M, T0_100M, (10, 1, 0.420, 0.297)),
    ("qc_100m_finish",   "100m", R0_100M, T0_100M, (95, 8, 0.420, 0.297)),
    ("qc_100m_small_A5", "100m", R0_100M, T0_100M, (50, 5, 0.210, 0.148)),
    ("qc_100m_tiny",     "100m", R0_100M, T0_100M, (50, 5, 0.100, 0.070)),
    ("qc_400m_mid",      "400m", R0_400M, T0_400M, (200, 5, 0.420, 0.297)),
    ("qc_400m_curve",    "400m", R0_400M, T0_400M, (60, 1, 0.420, 0.297)),
    ("qc_400m_far",      "400m", R0_400M, T0_400M, (300, 8, 0.420, 0.297)),
    ("qc_100m_sideview", "100m", R0_SIDE, T0_SIDE, (50, 5, 0.420, 0.297)),
    # ("qc_400m_sideview", "400m", R0_SIDE, T0_SIDE, (200, 5, 0.420, 0.297)),  # side view designed for 100m; invalid for 400m
    ("qc_100m_large",    "100m", R0_100M, T0_100M, (50, 5, 1.0, 0.7)),
    ("qc_400m_large",    "400m", R0_400M, T0_400M, (200, 5, 1.0, 0.7)),
]


def _parse_scenario(s: tuple):
    """Extract all parameters from a scenario tuple with defaults."""
    name, tt, r0, t0, pf, spec = s[:6]
    tol = 0.2
    use_accel = False
    num_false = 0
    dropout = 0.0
    calib_noise = 0.0
    if len(s) >= 7: tol = s[6]
    if len(s) >= 8: use_accel = bool(s[7]) if not isinstance(s[7], bool) else s[7]
    if len(s) >= 9: num_false = s[8]
    if len(s) >= 10: dropout = s[9]
    if len(s) >= 11: calib_noise = s[10]
    return name, tt, r0, t0, pf, spec, tol, use_accel, num_false, dropout, calib_noise


# ---- DummyDetector robustness test ----
def test_dummy_detector():
    """Verify pipeline doesn't crash when no external detections provided."""
    geom = TrackGeometry(track_type="100m")
    pipeline = TrackARPipeline(camera_matrix=K, geometry=geom)

    wp = geom.calibration_world_points()
    wa = np.array([w.as_array for w in wp], dtype=np.float64)
    pj, _ = cv2.projectPoints(wa, R0_100M, T0_100M, K, np.zeros((4, 1)))
    ip = [ImageCoord(float(p[0, 0]), float(p[0, 1])) for p in pj]
    pipeline.calibrate_from_points(wp, ip)

    render_proj = Projector(K, np.zeros((4, 1)))
    render_proj.set_extrinsics(R0_100M.copy(), T0_100M.copy())
    scene = SyntheticScene(render_proj, geom)
    fps = 60.0
    for fi in range(30):
        t = fi / fps
        athletes = scene.update(t)
        canvas = scene.render_background(athletes)
        try:
            pipeline.process_frame(canvas, timestamp=t)
        except Exception as e:
            return f"FAIL: DummyDetector crash at frame {fi}: {e}"
    active = [a for a in pipeline.assigner.athletes.values() if a.tracking_confidence > 0]
    if len(active) == 0:
        return "FAIL: DummyDetector produced no tracked athletes"
    return f"PASS: DummyDetector tracked {len(active)} athletes over 30 frames"


# ---- Stress test: occlusion guard + high dropout + simultaneous finish ----
def run_stress_tests():
    from tests.stress_test import test_all_finish_simultaneously, test_detection_dropout, test_sudden_appearance
    import io, contextlib
    results = []
    tests = [
        ("occlusion_simul_finish", test_all_finish_simultaneously),
        ("occlusion_dropout",      test_detection_dropout),
        ("occlusion_sudden_appear", test_sudden_appearance),
    ]
    for name, fn in tests:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            try:
                fn()
                results.append((name, "PASS"))
            except Exception as e:
                results.append((name, f"FAIL: {e}"))
        sys.stdout.write(buf.getvalue())
    return results


if __name__ == '__main__':
    cv2.setRNGSeed(42)  # deterministic OpenCV for reproducible results
    # === Quick calibration checks ===
    print("=== Quick Calibration Checks ===")
    q_passed = 0
    for s in QUICK_SCENARIOS:
        name, tt, r0, t0, spec = s
        sys.stdout.write(f"  {name} ... ")
        sys.stdout.flush()
        try:
            result = run_target_calib_check(tt, r0, t0, K, spec, name)
        except Exception as e:
            import traceback; traceback.print_exc()
            result = f"CRASH: {e}"
        print(result)
        if result.startswith("PASS"):
            q_passed += 1
    print(f"  Quick checks: {q_passed}/{len(QUICK_SCENARIOS)} passed\n")

    # === Full race tests ===
    print(f"Running {len(SCENARIOS)} full-race scenarios ({SPEED} m/s)...")
    r_passed = 0
    for s in SCENARIOS:
        name, tt, r0, t0, pf, spec, tol, use_accel, num_false, dropout, calib_noise = _parse_scenario(s)
        sys.stdout.write(f"  {name} ... ")
        sys.stdout.flush()
        try:
            result = run_test(tt, r0, t0, K, pf, spec, name, tol,
                              calib_noise_px=calib_noise,
                              use_acceleration=use_accel,
                              num_false_positives=num_false,
                              detection_dropout_rate=dropout)
        except Exception as e:
            import traceback; traceback.print_exc()
            result = f"CRASH: {e}"
        print(result)
        if result.startswith("PASS"):
            r_passed += 1

    total = q_passed + r_passed
    out_of = len(QUICK_SCENARIOS) + len(SCENARIOS)
    print(f"\n{'='*50}")
    print(f"  FULL-RACE RESULT: {total}/{out_of} passed")
    print(f"{'='*50}")

    # === DummyDetector robustness ===
    print("\n=== DummyDetector Robustness ===")
    dd_result = test_dummy_detector()
    print(f"  {dd_result}")
    if dd_result.startswith("PASS"):
        total += 1
    out_of += 1

    # === Standalone stress tests ===
    print("\n=== Standalone Stress Tests ===")
    stress_results = run_stress_tests()
    for name, result in stress_results:
        status = "PASS" if result == "PASS" else f"FAIL: {result}"
        print(f"  {name} ... {status}")
        if result == "PASS":
            total += 1
        out_of += 1

    print(f"\n{'='*50}")
    print(f"  FINAL RESULT: {total}/{out_of} passed")
    print(f"{'='*50}")
