import sys; sys.path.insert(0, '.')
import numpy as np, cv2, math, time
from calibration.coords import TrackGeometry, WorldCoord, ImageCoord
from calibration.projector import Projector
from pipeline.main_pipeline import TrackARPipeline
from tests.synthetic_scene import SyntheticScene
from tests.self_test import SCENARIOS, run_test, K

failing = [
    "100m/dropout50/std",
    "100m/static/noise3px",
]

for s in SCENARIOS:
    name = s[0]
    if name not in failing:
        continue
    sys.stdout.write(f'  {name} ... ')
    sys.stdout.flush()
    ts = time.time()
    try:
        tol = s[6] if len(s) >= 7 else 0.2
        use_accel = bool(s[7] if len(s) >= 8 else False) if isinstance(s[7] if len(s) >= 8 else False, (int, float)) else (s[7] if len(s) >= 8 else False)
        num_false = s[8] if len(s) >= 9 else 0
        dropout = s[9] if len(s) >= 10 else 0.0
        calib_noise = s[10] if len(s) >= 11 else 0.0
        result = run_test(s[1], s[2], s[3], K, s[4], s[5],
                          name=name, max_time_err_s=tol,
                          use_acceleration=use_accel,
                          num_false_positives=num_false,
                          detection_dropout_rate=dropout,
                          calib_noise_px=calib_noise)
        elapsed = time.time() - ts
    except Exception as e:
        import traceback; traceback.print_exc()
        result = f'CRASH: {e}'
        elapsed = time.time() - ts
    print(f'{result} ({elapsed:.0f}s)')
