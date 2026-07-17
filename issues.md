# TrackAR 问题清单

> 生成日期: 2026-07-16
> 来源: 全面代码审查（涉及全部 28 个源文件）

---

## 目录

- [CRITICAL（6 项）](#critical)
- [HIGH（8 项）](#high)
- [MEDIUM（14 项）](#medium)
- [LOW（15 项）](#low)

---

## CRITICAL

### C1. 400m 标定点世界坐标与 UI 指导不一致

**文件:** `calibration/coords.py:149-163`, `trackar_gui.py:17-24`, `run_real_video.py:160-167`

**问题:**
`calibration_world_points()` 对 400m 返回的 4 个世界坐标中，"Finish×Lane1" 和
"Finish×Lane8" 对应的物理位置是**对面直道尽头**（`arc=C+S`），不是终点线。
400m 中 Lane 1 的起点=终点（同一物理位置，`arc=0`），无法用标准"起终点 4 交点"
方式标定。当前代码改用对面直道尽头作为"远端点"是合理的，但：

- `trackar_gui.py:17` `CALIB_NAMES` 写的是 `"终点线×道1"` / `"终点线×道8"`
- `run_real_video.py:161-166` 提示用户点击 "Finish line x Lane 1/8"
- 用户按提示点击终点线 → PnP 得到错误外参 → 整个系统出错

**改正方向:**
- `calibration_world_points()` 应区分 100m/400m，400m 下不叫"Finish"
- 100m 保持现有行为（起终点各 4 点构成矩形）
- 400m 改名 `calibration_track_corners()` 或新增 400m 专用方法，返回
  [起点×道1, 起点×道8, 对面直道尽头×道1, 对面直道尽头×道8]
- GUI 和 CLI 的提示文字根据 track_type 动态切换
- 400m 标准标定模式下提示改为点击"对面直道尽头的两端"

---

### C2. `rects_overlap` 函数不存在但被导入

**文件:** `rendering/occlusion_guard.py`（缺少定义）, `tests/test_occlusion_guard.py:12`,
`tests/stress_test.py:11`

**问题:**
两个测试文件都从 `rendering.occlusion_guard` 导入 `rects_overlap`，但该模块中
从未定义此函数。运行两个测试文件都会在 import 阶段崩溃：

```python
# test_occlusion_guard.py:12
from rendering.occlusion_guard import OcclusionGuard, rects_overlap, compute_graphic_bbox
#                                                       ^^^^^^^^^^^^  ImportError!
```

这意味着：
- **遮挡防护单元测试全部是死代码**（7 个测试函数不能运行）
- **压力测试全部是死代码**（3 个测试函数不能运行）
- 测试覆盖率统计虚高

**改正方向:**
在 `occlusion_guard.py` 中实现 `rects_overlap` 函数：

```python
def rects_overlap(a: tuple, b: tuple, margin_px: float = 0) -> bool:
    x1 = max(a[0], b[0]) - margin_px
    y1 = max(a[1], b[1]) - margin_px
    x2 = min(a[2], b[2]) + margin_px
    y2 = min(a[3], b[3]) + margin_px
    return x1 < x2 and y1 < y2
```

---

### C3. `run_real_video.py` 跨帧标定代码无法运行

**文件:** `run_real_video.py:182-192`

**问题:**
三处致命错误：

1. `Line 182`: 传入了 `FrameTracker` 不存在的参数 `skip_interval=1`，构造函数只接受
   `max_width`。→ `TypeError`

2. `Line 185`: 访问 `ft.ref_kp`，但 `FrameTracker` 中该属性是私有的
   `self._ref_kp`（`frame_tracker.py:21`），没有 `ref_kp` property。
   → `AttributeError`

3. `Line 189`: `H = ft.update(gray)`，但 `FrameTracker.update()` 返回 `None`
   （void 方法）。`H` 始终为 `None`，后续代码未使用此值所以不崩溃，但语义完全错误。

**改正方向:**
```python
# 182 行: 移除不存在的参数
ft = FrameTracker(max_width=640)

# 185 行: 改为访问私有属性，或添加 public property
print(f"  Reference: {len(ft._ref_kp)} features detected")

# 189 行: 不赋值，update 是 void 方法
ft.update(gray)
# 后续的 current_to_calib 正确使用了内部的 H_calib_current，不需要 H 变量
```

---

### C4. 实际管线中 `track_homography` 只有 4 个 PnP 对应点

**文件:** `pipeline/main_pipeline.py:64`, `tests/self_test.py:92-93`

**问题:**
`main_pipeline.py` 的 `calibrate_from_points` 只存了 4 个标定点：

```python
self.projector.set_calibration_world_pts(world_pts)  # 只有 4 个点
```

而测试套件在标定后会额外添加稠密跟踪网格（~176-330 点）：

```python
track_pts = _add_tracking_grid(geom, calib_pts, target_spec)
pipeline.projector.set_calibration_world_pts(track_pts)
```

这意味着：**测试覆盖不到这个 bug**——测试中手动添加的稠密网格让 PnP 稳定，但
实际管线运行时 `track_homography` 只用了 4 个点。PnP 最小输入是 4 点，
对噪声极其敏感，相机运动稍大就会导致位姿抖动或发散。

**改正方向:**
- `calibrate_from_points` 中自动生成稠密跟踪网格，不要依赖调用方手动添加
- 在 `TrackGeometry` 或 `Projector` 中提供 `build_tracking_grid(step=10.0)` 方法
- 或直接在 `calibrate_from_points` 完成后自动调用一次网格生成

---

### C5. `trackar_gui.py` 后台线程访问 tkinter 变量

**文件:** `trackar_gui.py:234,245`

**问题:**
`_run_demo` 在后台线程（`threading.Thread`）中运行，但通过
`self.track_var.get()` 访问了 tkinter 的 `StringVar`。tkinter 变量
不是线程安全的，从非主线程访问可能导致：

- 程序崩溃（segfault）
- 数据损坏
- 平台相关的不确定行为

**改正方向:**
在启动线程前将需要的值从 tkinter 变量中读出，通过闭包或参数传递给线程：

```python
track_type = self.track_var.get()  # 主线程读取
thread = threading.Thread(target=self._run_demo, args=(track_type,), daemon=True)
```

---

### C6. `lane_assigner._build_image_cache` 缓存永不失效，400m 跟踪第 2 帧起全部错误

**文件:** `tracking/lane_assigner.py:99-128`, `calibration/lane_tracker.py:109-121`

**问题:**
`_build_image_cache` 使用 `id(self.projector)` 作为缓存键：

```python
cache_id = id(self.projector)
if getattr(self, '_img_cache_id', None) == cache_id:
    return  # 永不重建！
```

`id(projector)` 是对象内存地址，整个生命周期不变。但 `projector` 的外参
（`rvec`/`tvec`）每帧都在变——`track_homography` 每帧更新外参以跟踪相机运动。

后果：

1. **第 1 帧**：`_build_image_cache` 用当前外参构建 `_img_pts` 和 `_outer_hull`，正确
2. **第 2 帧起**：`track_homography` 更新了外参，但 `_build_image_cache` 检查缓存
   命中后直接返回——`_img_pts` 还是第 1 帧外参下的投影坐标
3. `_find_lane_dm_from_image` 用第 1 帧的投影坐标搜索第 N 帧的检测 → **车道和
   距离分配全部错误**
4. `_is_in_track_region` 用第 1 帧的 hull 做过滤 → 过滤边界完全错位

**为什么测试没发现？** 35 项测试中 400m 带相机运动的有 4 项
（`400m/pan/std`、`400m/zoom/std`、`400m/pan/tgt_mid`、`400m/zoom/tgt_mid`），
但扰动量极小（`_sin_amp` 振幅 0.0003 rad / 0.5m），第 1 帧的缓存近似有效，
误差落在时间容忍度（0.2s）内。真实摄像机大幅运动时必然崩溃。

`calibration/lane_tracker.py:109-121` 的 `_build_image_cache`（属于 `TrackGeometry`）
有同样问题，使用相同的 `id(projector)` 缓存策略（`coords.py:107`）。

**改正方向:**
```python
# 方案 A：每帧重建（简单，400m 模式下约 330 次投影，~1ms）
def _build_image_cache(self):
    if self.geometry._model is None:
        return
    # 移除缓存检查，每帧重建
    step = 2.0
    ...
```

```python
# 方案 B：缓存键包含外参版本号
_cache_key = (id(self.projector),
              tuple(self.projector.rvec.ravel()),
              tuple(self.projector.tvec.ravel()))
if getattr(self, '_img_cache_key', None) == _cache_key:
    return
```

---

## HIGH

### H1. 遮挡检测是空壳——未做实际碰撞检测

**文件:** `rendering/occlusion_guard.py:65-73`

**问题:**
`compute_safe_position` 遍历 `_PLACEMENT_CANDIDATES`，但**返回第一个世界坐标
在界内的候选**，从不检查图形是否与运动员检测框在图像平面重叠：

```python
for ahead_m, lateral_m, mode in _PLACEMENT_CANDIDATES:
    g_dm = d_m + (ahead_m if not mode.startswith("behind") else -ahead_m)
    if distance_to_end is not None and g_dm > self.geometry.length:
        continue  # 只检查了是否超出跑道范围
    if g_dm < 0:
        continue
    G_world = self.geometry.world_coord(lane, g_dm, lateral_shift=lateral_m, z=0.0)
    G_img = self.projector.project(G_world)
    return GraphicAnchor(...)  # 直接返回，没有碰撞检测！
```

名为"遮挡防护"，实际上完全没有实现核心功能。参数 `all_athletes` 传入但从未使用。

**改正方向:**
对每个候选位置：
1. 调用 `compute_graphic_bbox` 计算图形在图像空间的 bbox
2. 与该跑道运动员的检测框做碰撞检测（`rects_overlap`）
3. 也与其他跑道运动员做交叉碰撞检测
4. 有碰撞则尝试下一个候选，全部失败后走 fallback

---

### H2. 卡尔曼位置估计被架空

**文件:** `tracking/lane_assigner.py:339-340,395`

**问题:**
卡尔曼滤波更新后，位置被强置为测量值：

```python
athlete.kalman.update(np.array([dm]), confidence=athlete.tracking_confidence)
athlete.kalman.x[0, 0] = dm  # 覆盖卡尔曼的位置估计！
```

这样卡尔曼只保留了速度/加速度估计，位置部分完全没有发挥滤波作用（噪声抑制、
平滑、遮挡期间预测）。同样的代码出现在主匹配路径（line 340）和回收购路径
（line 395）。

**改正方向:**
移除 `kalman.x[0,0] = dm` 的覆盖，让卡尔曼的位置估计正常工作。
如果担心收敛速度，减小初始 P 矩阵或增大 process noise。

---

### H3. 400m 合成场景 `render_background` 多边形坐标错误

**文件:** `tests/synthetic_scene.py:165,168`

**问题:**
400m 渲染时，车道填充多边形和分界线的世界坐标使用了线性 d_m 值作为 x 坐标：

```python
fill = [WorldCoord(L * i / n, self.geometry.world_coord(lane, L * i / n).y, 0.0)
        for i in range(n + 1)]
```

对 400m 来说，d_m（沿跑道距离）和世界 x 坐标**不是线性关系**——弯道部分的
x 坐标由 `get_xy` 的三角函数计算。当前代码取世界坐标的 y 但用 d_m 替代 x，
导致：

- 车道多边形渲染到错误的图像位置
- 车道分界线位置错误
- KLT/ORB 跟踪器在错误的特征位置上匹配
- 对合成测试的可靠性有直接影响（虽然最终时间误差可能抵消）

**改正方向:**
使用 `get_xy` 返回的完整世界坐标：

```python
wc = self.geometry.world_coord(lane, L * i / n)
fill = [WorldCoord(wc.x, wc.y, 0.0)]
# 或直接使用 world_coord 返回的 WorldCoord
```

---

### H4. GUI 跨帧标定 ORB 在全分辨率下运行

**文件:** `trackar_gui.py:514-515`

**问题:**
```python
ft = FrameTracker(max_width=max(ref_gray.shape[1], 1920))  # 满分辨率！
ft.orb = cv2.ORB.create(nfeatures=8000, scaleFactor=1.5, nlevels=12)  # 8000 特征
```

在 1920×1080 全分辨率下提取 8000 个 ORB 特征，每帧需要 1-3 秒（CPU 计算）。
此代码在主线程中运行，导致 GUI 完全冻结数秒。

**改正方向:**
- 降采样到 1280 或 640 宽度
- 特征数降至 2000-4000
- 或在后台线程运行并显示进度

---

### H5. CSV 日志文件句柄泄漏

**文件:** `utils/logger.py:30,45`

**问题:**
`start_session` 打开文件但从不关闭：

```python
f = open(self.csv_file, 'w', newline='')  # 文件已打开
self.csv_writer = csv.writer(f)
# ... 文件句柄 f 从未保存到 self
```

`close()` 方法是空实现：

```python
def close(self):
    pass
```

Windows 上文件句柄不释放会导致文件锁定，且未 flush 的缓冲区可能丢失数据。

**改正方向:**
```python
def start_session(self, ...):
    self._csv_file = open(...)
    self.csv_writer = csv.writer(self._csv_file)

def log(self, record):
    if self.csv_writer:
        self.csv_writer.writerow([...])
        self._csv_file.flush()  # 实时写入

def close(self):
    if hasattr(self, '_csv_file') and self._csv_file:
        self._csv_file.flush()
        self._csv_file.close()
        self._csv_file = None
```

---

### H6. 100m 合成场景运动员不停止在终点

**文件:** `tests/synthetic_scene.py:30,48`

**问题:**
100m 模式下 `_finish_dms` 是空 dict（line 30），因为 `_model` 为 None
导致 line 31-34 的填充循环不会执行。line 48 的检查永远为 False：

```python
if self._finish_dms and finished:  # _finish_dms = {} → False
    d_m = self._finish_dms[lane]   # 永不执行
```

100m 运动员的 d_m 会继续增长超过 100m，影响 `render()`（demo.py 和 GUI 演示）。

**改正方向:**
在 `__init__` 中统一处理：
```python
for lane in range(1, 9):
    self._finish_dms[lane] = self.length  # 100m 和 400m 都设置
```

---

### H7. 400m 合成场景 `render()` 同样坐标错误

**文件:** `tests/synthetic_scene.py:77-82`

**问题:**
与 H3 相同，但出现在旧的 `render()` 方法中（用于 `demo.py` 和 GUI 演示）。
这里的 400m 车道填充多边形也用了 `L*i/n` 作为 x 坐标。

**改正方向:**
与 H3 相同。

---

### H8. `media_io/video_io.py` 竞态条件和资源泄漏

**文件:** `media_io/video_io.py:26,38`

**问题:**
1. `open()` 被多次调用时，前一个 `VideoCapture` 不被释放（line 26）
2. `_read_loop` 中 `self.cap.read()` 没有锁保护，`stop()` 可能并发释放 cap
   （line 38-44）

**改正方向:**
```python
def open(self, source):
    if self.cap:
        self.cap.release()
    self.cap = cv2.VideoCapture(uri)
    ...
```

在 `_read_loop` 中加锁访问 `self.cap`。

---

## MEDIUM

### M1. `FrameTracker` 死代码

**文件:** `calibration/frame_tracker.py:3,25-26`

**问题:**
- `from concurrent.futures import ThreadPoolExecutor` 导入但从未使用
- `self._executor = ThreadPoolExecutor(max_workers=2)` 创建了两个永久空转的线程
- `self._pending_feature` 初始化后从未读写（死代码）
- `_filter_matches` 方法定义在 line 70 但从未被调用（Lowe ratio test 未集成）

**改正方向:** 移除这些死代码和空转线程池。

---

### M2. `DepthSorter` 整个文件是死代码

**文件:** `rendering/depth_sorter.py`（全部 32 行）

**问题:**
`DepthSorter` 类和 `DepthLayer` dataclass 已定义但整个项目没有一处导入或使用。
README 和设计文档（`track-ar-system-design.md:185,322`）中提到了深度排序，
但实际渲染管线完全跳过了它。

**改正方向:** 要么实现深度排序并集成到渲染管线，要么删除文件和相关文档。

---

### M3. 起跑检测逻辑脆弱

**文件:** `pipeline/main_pipeline.py:142-148`

**问题:**
起跑检测使用硬编码阈值和固定窗口：

```python
past_threshold = sum(1 for p in positions if p.d_m > 1.0 and p.speed_mps > 2.0
                     and p.confidence > 0.0)
self._start_decisions.append(past_threshold)
if len(self._start_decisions) > 3:
    self._start_decisions.pop(0)
consistent = sum(1 for v in self._start_decisions if v >= 3) >= 2
if consistent:
    self.timer.start_race(timestamp)
```

- 阈值 `d_m > 1.0` 对不同镜头距离不通用
- `speed_mps > 2.0` 在低帧率下不稳定
- `≥3` 人的硬编码值不适合少于 8 名运动员的场景
- 滚动窗口大小 3 固定

**改正方向:**
- 将阈值参数化，允许根据 track_type 和 fps 自适应
- 考虑使用更鲁棒的起跑信号（如 ≥2 名运动员的 d_m 同时开始增长超过噪声水平）

---

### M4. `EdgeCaseDetector` 一半方法未集成

**文件:** `pipeline/edge_cases.py:36,54`

**问题:**
`check_speed_anomaly` 和 `check_lane_switch` 都已实现（含历史记录维护），
但 `check_all()` 只调用了 `check_fallen` 和 `check_finish_line`：
- `check_speed_anomaly` 从未被调用，但 `speed_history` 会无限增长
- `check_lane_switch` 从未被调用
- `alert_history` 定义但从未使用

**改正方向:** 在 `check_all` 中集成这些检查，或标记为供将来使用并抑制警告。

---

### M5. `Preprocessor.enable_undistortion` 未实现

**文件:** `pipeline/preprocessor.py:15-17`

**问题:**
构造函数接受 `enable_undistortion` 参数并赋值给 `self.enable_undistortion`，
但从未在任何地方检查或使用该标志。镜头畸变校正从未实现。

**改正方向:** 移除参数或实现实际的畸变校正（使用 `projector.dist_coeffs`）。

---

### M6. 卡尔曼速度钳制只发生在 update 后

**文件:** `tracking/kalman.py:52,74`

**问题:**
速度钳制（±15 m/s）只在 `update()` 方法中执行（line 74）。
`predict()` 方法（line 52）可能因加速度累积产生超范围速度。
如果连续多帧没有 measurement update（coast 模式），速度可能发散。

**改正方向:**
在 `predict()` 中也钳制速度：

```python
def predict(self):
    if not self.initialized:
        return
    self.x = self.F @ self.x
    self.x[1, 0] = np.clip(self.x[1, 0], -15.0, 15.0)  # 预测后钳制速度
    self.x[0, 0] = max(0.0, self.x[0, 0])  # 预测后钳制位置
    self.P = self.F @ self.P @ self.F.T + self.Q
```

---

### M7. `track_homography` 返回值语义误导

**文件:** `calibration/projector.py:22,37-38`

**问题:**
文档字符串说 "Returns True if pose was updated"，但当位移小于阈值时也返回
`True`（line 37-38），实际上没有做任何更新：

```python
if displacement < min_displacement_px:
    return True  # 返回 True 但位姿没变！
```

调用方可能误以为位姿已更新。同时，`min_displacement_px` 默认值为 0.0，使
这个提前返回路径永远不会触发（`0 < 0` 为 False），形同虚设。

**改正方向:**
- 明确语义：只有真正调用 `solvePnP` 并成功更新时才返回 True
- 无更新时返回 False
- 或将 `min_displacement_px` 改为正数使其生效

---

### M8. 卡尔曼创新门控硬拒绝

**文件:** `tracking/kalman.py:70-71`

**问题:**
```python
if abs(innovation) > 3.0 * sigma:
    return  # 完全丢弃该测量
```

超过 3σ 的测量被完全忽略。在快速相机运动、遮挡恢复或 YOLO 检测跳跃时，
连续多帧的被拒绝会导致卡尔曼发散。

**改正方向:**
使用软门控：根据创新大小自适应放大测量噪声（R），而不是硬丢弃：

```python
innovation_mahal = abs(innovation) / sigma
if innovation_mahal > 3.0:
    R = R * (innovation_mahal / 3.0)  # 自适应放大噪声
    # 继续更新
```

---

### M9. `run_on_video` / `run_live` 缺少 try/finally

**文件:** `pipeline/main_pipeline.py:213,244`

**问题:**
`run_on_video` 和 `run_live` 在处理循环中可能抛出异常（如 `process_frame`
内部错误），此时 `cap.release()` 和 `writer.release()` 不会执行。
VideoCapture 和 VideoWriter 资源泄漏。

**改正方向:**
```python
cap = cv2.VideoCapture(video_path)
writer = None
try:
    # ... 处理循环 ...
finally:
    cap.release()
    if writer:
        writer.release()
```

---

### M10. `DynamicCamera.update` 修改共享的可变状态

**文件:** `pipeline/dynamic_camera.py:42`

**问题:**
`DynamicCamera.update` 调用 `self.projector.look_at()` 修改 `Projector` 的
外参。但同一个 `Projector` 实例被多个 pipeline stage 共享（遮挡防护、贴花渲染、
调试叠加）。如果在 `process_frame` 中途切换 `follow_mode`，行为未定义。

**改正方向:**
- 为 `DynamicCamera` 使用独立的 `Projector` 副本，或
- 文档说明 `follow_mode` 只应在帧间切换，或
- 将投影和渲染分离为两个阶段

---

### M11. `RaceTimer` 完成时间可能为负

**文件:** `pipeline/timing.py:20-21,35-36`

**问题:**
`finish_race(None)` 时时间设为 0.0：

```python
def finish_race(self, timestamp: float | None = None):
    if not self.race_finished:
        self.finish_time = timestamp if timestamp is not None else 0.0
```

此时 `get_elapsed()` 返回 `0.0 - t0` 即负数。

**改正方向:**
```python
def finish_race(self, timestamp: float | None = None):
    if not self.race_finished:
        self.finish_time = timestamp if timestamp is not None else self.t0
        self.race_finished = True
```

---

### M12. `render()` 方法在动态相机下仍然存在

**文件:** `demo.py:76`, `trackar_gui.py:288`

**问题:**
`demo.py` 和 `trackar_gui._run_demo` 中合成场景使用 `scene.render(athletes)` 绘制
画面。这个方法（`synthetic_scene.py:57`）绘制粗圆圈代表运动员，并使用
`projector.project()` 定位。但 `render()` 接受的是 `SynthAthleteState` 列表而不是
`np.ndarray` 图像，且不从 `render_background()` 继承纹理。

`render()` 和 `render_background()` 是两套独立的渲染路径：
- `render()` → 有醒目运动员标记，用于人眼观看的 demo
- `render_background()` → 有逼真纹理，用于测试 KLT/ORB 跟踪

但在合成场景中，YOLO/dummy 检测器接收的输出是 `render()` 的画面。当 `follow_mode`
启用时，`DynamicCamera` 每帧修改 `projector` 外参，但 `render()` 使用的
`projector` 与管线共用同一实例。这会导致：

1. `DynamicCamera.update(positions)` 修改外参 → 下一帧的 `render()` 在新外参下绘制
2. 但 `frame_tracker.update(gray)` 接收的画面已是新视角 → 累积 H 不正确

**改正方向:**
```python
# demo.py 中使用独立 render_proj 做渲染，pipeline 用另一个 projector 跟踪
render_proj = Projector(K)
render_proj.set_extrinsics(rvec.copy(), tvec.copy())
scene = SyntheticScene(render_proj, geom, speeds=speeds)
# 动态相机应该修改 render_proj，而不是 pipeline.projector
```

---

### M13. `position_estimator.py:41` 负帧差未保护

**文件:** `tracking/position_estimator.py:41-43`

```python
if prev is not None:
    dt = timestamp - prev.timestamp
    if dt > 0:
        speed = (raw_dm - prev.d_m) / dt
```

当 `dt <= 0`（时间戳倒退或相同）时 `speed` 保持为 0.0，但 `raw_dm` 未被钳制。
时间戳倒退时 `jump` 计算仍然进行（line 48），可能导致误判为跳跃。

**改正方向:** 在 `dt <= 0` 时 `continue` 跳过整段速度/跳跃检测。

---

### M14. `PositionEstimator` 起跑阶段跳跃钳制无效

**文件:** `tracking/position_estimator.py:49`

**问题:**
```python
if jump > self.MAX_JUMP_M and prev.d_m > 1.0:  # d_m>1 才钳制
```

起跑的前几帧（d_m < 1.0）跳变不会被钳制，此时运动员从 0 突然跳到 ~0.3m
（实际是正常起步），虽然不算 bug 但逻辑不一致。

**改正方向:**
使用帧计数而非 d_m 阈值：

```python
if jump > self.MAX_JUMP_M and self.frame_count > 10:
```

---

## LOW

### L1. `lane_assigner.py` 方法体内重复 import cv2

**文件:** `tracking/lane_assigner.py:123`

模块级已有 `import cv2`（line 1），`_build_image_cache` 方法内又导入了
一次。

**改正方向:** 移除 line 123 的 `import cv2`。

---

### L2. `coords.py` 方法体内无用 import

**文件:** `calibration/coords.py:99`

```python
from calibration.projector import Projector
```

导入后从未使用。参数 `projector` 已经传入，类型已经在模块级可知。

**改正方向:** 移除该 import。

---

### L3. `main_pipeline.py` 未使用的变量

**文件:** `pipeline/main_pipeline.py:123-131`

`saved_r` 和 `saved_t` 赋值后从未被读取。后续的注释说明这是意图恢复的，
但实际没有恢复操作。

**改正方向:** 移除赋值，或实现完整的 save/restore 逻辑。

---

### L4. `finish_distances` 每帧重建

**文件:** `pipeline/main_pipeline.py:199`

```python
finish_distances = {lane: self.geometry.finish_distance(lane) for lane in range(1, 9)}
```

该字典是静态的（每帧相同），但每帧重建一次。

**改正方向:** 在 `__init__` 或 `calibrate_from_points` 中计算一次并缓存。

---

### L5. `graphic_factory.py` 无意义的 addWeighted

**文件:** `rendering/graphic_factory.py:37`

```python
canvas = cv2.addWeighted(canvas, 0, overlay, 1, 0)
```

`canvas` 是 `np.zeros()`，等价于 `canvas = overlay.copy()`。

**改正方向:** 简化为 `canvas = overlay.copy()` 或直接 `return overlay`。

---

### L6. IoU 分母使用 min() 令人困惑

**文件:** `tracking/lane_assigner.py:312`

```python
iou = inter / min(ai_area + aj_area - inter, ai_area + aj_area)
```

由于 `inter ≥ 0`，`min()` 始终返回 `ai_area + aj_area - inter`（即标准并集），
min() 是多余操作。

**改正方向:** 简化为 `iou = inter / (ai_area + aj_area - inter + 1e-6)`。

---

### L7. 不可达的分支和检查

**文件:**
- `tracking/lane_assigner.py:172`: `if lane is None: lane = 1` — `lane_from_y` 永不返回 None
- `tracking/lane_assigner.py:369-372`: 400m 和 100m 分支调用相同的函数，
  `dm > length*2` 检查因函数内部已 clip 而不可达
- `pipeline/main_pipeline.py:132`: `if not self.frame_tracker.is_ready()` —
  刚调用 update 后永远为 True

**改正方向:** 移除不可达代码。

---

### L8. `calibrator.py` 4 点 PnP 使用 RANSAC 无意义

**文件:** `calibration/calibrator.py:29`

对 4 个标定点使用 `solvePnPRansac`。RANSAC 需要至少 4 个点才能采样，对 4 点
来说 RANSAC 退化为普通 solvePnP，额外引入计算开销。

**改正方向:** `if len(world_pts) == 4: use solvePnP directly`。

---

### L9. GUI 路径字符串 shell 转义风险

**文件:** `trackar_gui.py:693`

```python
subprocess.Popen(f'explorer /select,"{path}"')
```

路径含空格和特殊字符时可能被错误解析。

**改正方向:**
```python
subprocess.Popen(['explorer', '/select,', path])
```

---

### L10. 所有参数硬编码，无配置文件

**覆盖范围:** 整个项目

几乎所有阈值、窗口大小、容忍度都是 Python 类常量或 magic number。
如 `lane_assigner.py` 的 `MARGIN_PX=20`、`MATCH_MAX_PX_DIST=200.0`、
`MIN_ATHLETE_HEIGHT=60`、`MAX_DM_JUMP=10.0` 等。

**改正方向:** 引入 YAML/JSON 配置文件，使用 `@dataclass` 配置对象加载。

---

### L11. YOLO 模型不缓存

**文件:** `detection/detector.py:57-65`

`YOLODetector.__init__` 每次都从磁盘加载模型权重。如果多次创建 detector
实例（如重置时），会重复加载。

**改正方向:** 实现模型缓存（module-level cache 或 singleton）。

---

### L12. `runner_real_video.py` 进度报告浮点精度

**文件:** `run_real_video.py:261`

```python
pct = frame_idx / min(args.max_frames, total_frames) * 100
```

当 `max_frames=-1`（默认值）时，`min(-1, total_frames)` 返回 -1，导致除零。

**改正方向:**
```python
n_total = min(args.max_frames, total_frames) if args.max_frames > 0 else total_frames
pct = frame_idx / n_total * 100
```

---

### L13. `control_panel.py` 重置后相机位置硬编码

**文件:** `ui/control_panel.py:95,100`

```python
self.pipeline.projector.look_at(WorldCoord(50.0, 4.88, 0.0))
```

`50.0` 只适用于 100m 赛道。对 400m 赛道应指向 `self.pipeline.geometry.length / 2`。

**改正方向:** 使用 `self.pipeline.geometry.length / 2` 和 `lane_center_y(4.5)`。

---

### L14. `scripts/p4_integration_test.py` 死代码

**文件:** `scripts/p4_integration_test.py:56-59`

```python
pipeline = TrackARPipeline()          # 立即被下一行覆盖
geom = TrackGeometry()                # 定义后从未使用
pipeline = TrackARPipeline(camera_matrix=K)  # 正确的初始化
```

第一行 `TrackARPipeline()` 创建了完整的 pipeline（含 projector、assigner、detector
等全部子对象），然后立即丢弃。`geom` 变量也未被使用。浪费内存和构造时间。

**改正方向:** 移除无用的第一行和 `geom` 变量。

---

### L15. `calibration_world_points` 返回的 `finish_distance` 与标定点不一致

**文件:** `calibration/coords.py:144-147,149-163`

`finish_distance()` 返回 400.0（比赛距离），但 `calibration_world_points` 的
"Finish" 点对应的是 d_m=200（对面直道尽头）。虽然逻辑正确（finish 检测确实在
d_m=400），但命名和 API 设计不一致，容易导致维护者误解。

**改正方向:** 统一术语——标定点称为 "track corners" 而非 "start/finish"。

---

## 统计

| 严重级别 | 数量 |
|----------|------|
| CRITICAL | 6 |
| HIGH     | 8 |
| MEDIUM   | 14 |
| LOW      | 15 |
| **总计** | **43** |
