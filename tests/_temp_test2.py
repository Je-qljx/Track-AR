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
render_proj = Projector(K, np.zeros((4,1)))
render_proj.set_extrinsics(r0.copy(), t0.copy())
scene = SyntheticScene(render_proj, geom, speeds=[9.5]*8)

canvas = scene.render_background([])
gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
print(f'Image: {gray.shape}, min={gray.min()}, max={gray.max()}, mean={gray.mean():.1f}')

# Test feature detection at various quality levels
for ql in [0.005, 0.001, 0.0005, 0.0001]:
    pts = cv2.goodFeaturesToTrack(gray, maxCorners=600, qualityLevel=ql, minDistance=2.0, blockSize=5)
    n = len(pts) if pts is not None else 0
    print(f'  qualityLevel={ql}: {n} features')

# Test ORB
orb = cv2.ORB.create(nfeatures=1200, scaleFactor=1.2, nlevels=8)
kp, des = orb.detectAndCompute(canvas, None)
print(f'ORB: {len(kp)} features')

# Test finding homography from these
# Try different resolutions
for scale_name, w in [('640', 640), ('960', 960), ('1280', 1280)]:
    lr = cv2.resize(gray, (w, int(gray.shape[0] * w / gray.shape[1])))
    pts = cv2.goodFeaturesToTrack(lr, maxCorners=600, qualityLevel=0.001, minDistance=2.0, blockSize=5)
    n = len(pts) if pts is not None else 0
    print(f'  {scale_name}x{lr.shape[0]}: qualityLevel=0.001 -> {n} features')
