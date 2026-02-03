# ROI Selection Feature Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add region-of-interest (ROI) selection step after video upload where users draw a bounding box around the fretboard and left hand.

**Architecture:** Frontend gains a new `ROISelector` component shown between upload and processing. ROI coordinates (normalized 0-1) are passed to backend via POST /jobs, then used to crop frames before MediaPipe and fretboard detection.

**Tech Stack:** React + Zustand (frontend), Flask + OpenCV (backend)

---

### Task 1: Add ROI fields to backend Job model

**Files:**
- Modify: `tabvision-server/app/models.py:8-45`
- Test: `tabvision-server/tests/test_models.py`

**Step 1: Write the failing test**

Add to `tabvision-server/tests/test_models.py`:

```python
def test_job_with_roi_fields():
    """Job can be created with ROI coordinates."""
    job = Job.create(video_path="/test.mp4", capo_fret=0)
    job.roi_x1 = 0.1
    job.roi_y1 = 0.2
    job.roi_x2 = 0.8
    job.roi_y2 = 0.9

    assert job.roi_x1 == 0.1
    assert job.roi_y1 == 0.2
    assert job.roi_x2 == 0.8
    assert job.roi_y2 == 0.9


def test_job_roi_defaults_to_none():
    """Job ROI fields default to None."""
    job = Job.create(video_path="/test.mp4", capo_fret=0)

    assert job.roi_x1 is None
    assert job.roi_y1 is None
    assert job.roi_x2 is None
    assert job.roi_y2 is None
```

**Step 2: Run test to verify it fails**

Run: `cd tabvision-server && python -m pytest tests/test_models.py::test_job_with_roi_fields -v`
Expected: FAIL with "AttributeError: 'Job' object has no attribute 'roi_x1'"

**Step 3: Write minimal implementation**

Modify `tabvision-server/app/models.py` - add ROI fields to Job dataclass after `error_message`:

```python
@dataclass
class Job:
    id: str
    status: str  # pending | processing | completed | failed
    created_at: datetime
    updated_at: datetime
    video_path: str
    capo_fret: int
    progress: float
    current_stage: str
    result_path: Optional[str] = None
    error_message: Optional[str] = None
    # ROI coordinates (normalized 0-1)
    roi_x1: Optional[float] = None
    roi_y1: Optional[float] = None
    roi_x2: Optional[float] = None
    roi_y2: Optional[float] = None
```

**Step 4: Run test to verify it passes**

Run: `cd tabvision-server && python -m pytest tests/test_models.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tabvision-server/app/models.py tabvision-server/tests/test_models.py
git commit -m "$(cat <<'EOF'
feat(backend): add ROI fields to Job model

Add roi_x1, roi_y1, roi_x2, roi_y2 optional fields to Job dataclass
for storing user-defined region of interest coordinates.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Accept ROI parameters in POST /jobs route

**Files:**
- Modify: `tabvision-server/app/routes.py:19-64`
- Test: `tabvision-server/tests/test_routes.py`

**Step 1: Write the failing test**

Add to `tabvision-server/tests/test_routes.py`:

```python
def test_post_jobs_with_roi(client):
    """POST /jobs accepts ROI coordinates."""
    with patch('app.routes.Thread') as mock_thread:
        data = {
            'video': (io.BytesIO(b'fake video content'), 'test.mp4'),
            'capo_fret': '0',
            'roi_x1': '0.1',
            'roi_y1': '0.2',
            'roi_x2': '0.8',
            'roi_y2': '0.9',
        }
        response = client.post('/jobs', data=data, content_type='multipart/form-data')

        assert response.status_code == 201
        job_id = response.get_json()['job_id']

        # Verify ROI was stored in job
        job = job_storage.get(job_id)
        assert job.roi_x1 == 0.1
        assert job.roi_y1 == 0.2
        assert job.roi_x2 == 0.8
        assert job.roi_y2 == 0.9


def test_post_jobs_validates_roi_range(client):
    """POST /jobs validates ROI coordinates are in 0-1 range."""
    with patch('app.routes.Thread'):
        data = {
            'video': (io.BytesIO(b'fake video content'), 'test.mp4'),
            'capo_fret': '0',
            'roi_x1': '0.1',
            'roi_y1': '0.2',
            'roi_x2': '1.5',  # Invalid: > 1
            'roi_y2': '0.9',
        }
        response = client.post('/jobs', data=data, content_type='multipart/form-data')

        assert response.status_code == 400
        assert 'ROI' in response.get_json()['error']


def test_post_jobs_validates_roi_order(client):
    """POST /jobs validates x1 < x2 and y1 < y2."""
    with patch('app.routes.Thread'):
        data = {
            'video': (io.BytesIO(b'fake video content'), 'test.mp4'),
            'capo_fret': '0',
            'roi_x1': '0.8',  # Invalid: x1 > x2
            'roi_y1': '0.2',
            'roi_x2': '0.1',
            'roi_y2': '0.9',
        }
        response = client.post('/jobs', data=data, content_type='multipart/form-data')

        assert response.status_code == 400
        assert 'ROI' in response.get_json()['error']
```

**Step 2: Run test to verify it fails**

Run: `cd tabvision-server && python -m pytest tests/test_routes.py::test_post_jobs_with_roi -v`
Expected: FAIL with "assert job.roi_x1 == 0.1" (roi_x1 is None)

**Step 3: Write minimal implementation**

Modify `tabvision-server/app/routes.py` - update `create_job()` function:

```python
@bp.route('/jobs', methods=['POST'])
def create_job():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed. Use MP4 or MOV.'}), 400

    capo_fret = request.form.get('capo_fret', '0')
    try:
        capo_fret = int(capo_fret)
    except ValueError:
        capo_fret = 0

    # Parse ROI coordinates
    roi_x1 = request.form.get('roi_x1')
    roi_y1 = request.form.get('roi_y1')
    roi_x2 = request.form.get('roi_x2')
    roi_y2 = request.form.get('roi_y2')

    # Validate ROI if provided
    roi = None
    if all(v is not None for v in [roi_x1, roi_y1, roi_x2, roi_y2]):
        try:
            roi = {
                'x1': float(roi_x1),
                'y1': float(roi_y1),
                'x2': float(roi_x2),
                'y2': float(roi_y2),
            }
            # Validate range
            for key, val in roi.items():
                if not 0 <= val <= 1:
                    return jsonify({'error': f'ROI {key} must be between 0 and 1'}), 400
            # Validate order
            if roi['x1'] >= roi['x2']:
                return jsonify({'error': 'ROI x1 must be less than x2'}), 400
            if roi['y1'] >= roi['y2']:
                return jsonify({'error': 'ROI y1 must be less than y2'}), 400
        except ValueError:
            return jsonify({'error': 'ROI coordinates must be valid numbers'}), 400

    # Save the file
    filename = secure_filename(file.filename)
    upload_folder = current_app.config['UPLOAD_FOLDER']
    os.makedirs(upload_folder, exist_ok=True)

    # Create job first to get ID for unique filename
    job = Job.create(video_path="", capo_fret=capo_fret)

    # Set ROI if provided
    if roi:
        job.roi_x1 = roi['x1']
        job.roi_y1 = roi['y1']
        job.roi_x2 = roi['x2']
        job.roi_y2 = roi['y2']

    # Use job ID in filename to ensure uniqueness
    ext = filename.rsplit('.', 1)[1].lower()
    unique_filename = f"{job.id}.{ext}"
    file_path = os.path.join(upload_folder, unique_filename)
    file.save(file_path)

    # Update job with file path
    job.video_path = file_path
    job_storage.save(job)

    # Launch background processing
    results_folder = current_app.config.get('RESULTS_FOLDER', upload_folder)
    thread = Thread(
        target=process_job,
        args=(job.id, job_storage, results_folder),
        daemon=True
    )
    thread.start()

    return jsonify({'job_id': job.id}), 201
```

**Step 4: Run test to verify it passes**

Run: `cd tabvision-server && python -m pytest tests/test_routes.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tabvision-server/app/routes.py tabvision-server/tests/test_routes.py
git commit -m "$(cat <<'EOF'
feat(backend): accept ROI parameters in POST /jobs

Parse and validate roi_x1, roi_y1, roi_x2, roi_y2 from form data.
Validates coordinates are in 0-1 range and x1 < x2, y1 < y2.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Add ROI cropping to video frame extraction

**Files:**
- Modify: `tabvision-server/app/video_pipeline.py:129-162`
- Test: `tabvision-server/tests/test_video_pipeline.py`

**Step 1: Write the failing test**

Add to `tabvision-server/tests/test_video_pipeline.py`:

```python
class TestExtractFrameWithROI:
    """Tests for frame extraction with ROI cropping."""

    def test_extract_frame_with_roi_crops_correctly(self, tmp_path):
        """extract_frame with ROI returns cropped frame."""
        import cv2

        # Create a test video with known dimensions
        video_path = str(tmp_path / "test.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 30, (100, 100))

        # Create frame with quadrant colors for easy verification
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[0:50, 0:50] = (255, 0, 0)    # Top-left: blue
        frame[0:50, 50:100] = (0, 255, 0)  # Top-right: green
        frame[50:100, 0:50] = (0, 0, 255)  # Bottom-left: red
        frame[50:100, 50:100] = (255, 255, 0)  # Bottom-right: cyan
        for _ in range(30):
            out.write(frame)
        out.release()

        # Extract with ROI covering top-left quadrant
        roi = {'x1': 0.0, 'y1': 0.0, 'x2': 0.5, 'y2': 0.5}
        result = extract_frame(video_path, 0.5, roi=roi)

        assert result is not None
        assert result.shape == (50, 50, 3)
        # Should be blue (BGR)
        assert result[25, 25, 0] == 255  # Blue channel
        assert result[25, 25, 1] == 0    # Green channel
        assert result[25, 25, 2] == 0    # Red channel

    def test_extract_frame_without_roi_returns_full_frame(self, tmp_path):
        """extract_frame without ROI returns full frame."""
        import cv2

        video_path = str(tmp_path / "test.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 30, (100, 100))
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        for _ in range(30):
            out.write(frame)
        out.release()

        result = extract_frame(video_path, 0.5)

        assert result is not None
        assert result.shape == (100, 100, 3)
```

**Step 2: Run test to verify it fails**

Run: `cd tabvision-server && python -m pytest tests/test_video_pipeline.py::TestExtractFrameWithROI -v`
Expected: FAIL with "TypeError: extract_frame() got an unexpected keyword argument 'roi'"

**Step 3: Write minimal implementation**

Modify `tabvision-server/app/video_pipeline.py` - update `extract_frame()`:

```python
def extract_frame(
    video_path: str,
    timestamp: float,
    roi: dict = None
) -> np.ndarray | None:
    """Extract a single frame from video at given timestamp.

    Args:
        video_path: Path to video file
        timestamp: Time in seconds to extract frame at
        roi: Optional dict with x1, y1, x2, y2 (normalized 0-1) to crop frame

    Returns:
        BGR numpy array of the frame, or None if extraction fails
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    # Get video FPS to calculate frame number
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return None

    # Calculate target frame number
    frame_number = int(timestamp * fps)

    # Seek to the target frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return None

    # Apply ROI cropping if provided
    if roi is not None:
        h, w = frame.shape[:2]
        x1 = int(roi['x1'] * w)
        y1 = int(roi['y1'] * h)
        x2 = int(roi['x2'] * w)
        y2 = int(roi['y2'] * h)
        frame = frame[y1:y2, x1:x2]

    return frame
```

**Step 4: Run test to verify it passes**

Run: `cd tabvision-server && python -m pytest tests/test_video_pipeline.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tabvision-server/app/video_pipeline.py tabvision-server/tests/test_video_pipeline.py
git commit -m "$(cat <<'EOF'
feat(backend): add ROI cropping to frame extraction

extract_frame() now accepts optional roi dict to crop frames
before returning. Coordinates are normalized 0-1.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Pass ROI through video analysis pipeline

**Files:**
- Modify: `tabvision-server/app/video_pipeline.py:362-449`
- Test: `tabvision-server/tests/test_video_pipeline.py`

**Step 1: Write the failing test**

Add to `tabvision-server/tests/test_video_pipeline.py`:

```python
def test_analyze_video_at_timestamps_with_roi(tmp_path):
    """analyze_video_at_timestamps passes ROI to frame extraction."""
    import cv2

    video_path = str(tmp_path / "test.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 30, (100, 100))
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    for _ in range(60):
        out.write(frame)
    out.release()

    roi = {'x1': 0.25, 'y1': 0.25, 'x2': 0.75, 'y2': 0.75}

    # Track what frames are passed to detect_hand_landmarks
    captured_frames = []

    def mock_detect(frame, config=None, landmarker=None):
        captured_frames.append(frame.shape)
        return HandObservation(
            timestamp=0.0,
            fingers=[FingerPosition(i, 0.5, 0.5, 0.0) for i in range(5)],
            is_left_hand=True,
        )

    with patch('app.video_pipeline.detect_hand_landmarks', side_effect=mock_detect):
        with patch('app.video_pipeline._get_hand_landmarker'):
            result = analyze_video_at_timestamps(video_path, [0.5], roi=roi)

    # Cropped frame should be 50x50 (half of 100x100)
    assert len(captured_frames) > 0
    assert captured_frames[0] == (50, 50, 3)
```

**Step 2: Run test to verify it fails**

Run: `cd tabvision-server && python -m pytest tests/test_video_pipeline.py::test_analyze_video_at_timestamps_with_roi -v`
Expected: FAIL with "TypeError: analyze_video_at_timestamps() got an unexpected keyword argument 'roi'"

**Step 3: Write minimal implementation**

Modify `tabvision-server/app/video_pipeline.py` - update `analyze_video_at_timestamps()`:

```python
def analyze_video_at_timestamps(
    video_path: str,
    timestamps: list[float],
    config: Optional[VideoAnalysisConfig] = None,
    roi: dict = None
) -> dict[float, HandObservation]:
    """Analyze video frames at given timestamps.

    Extracts frames at each timestamp and runs hand detection.
    Uses multiple frames per onset for better accuracy.

    Args:
        video_path: Path to video file
        timestamps: List of timestamps (in seconds) to analyze
        config: Analysis configuration
        roi: Optional dict with x1, y1, x2, y2 (normalized 0-1) to crop frames

    Returns:
        Dictionary mapping timestamp to HandObservation
        (only includes timestamps where a hand was detected)
    """
    if config is None:
        config = VideoAnalysisConfig()

    observations = {}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return observations

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return observations

    # Get frame dimensions for ROI calculation
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a single landmarker instance for all frames (more efficient)
    try:
        landmarker = _get_hand_landmarker(config)
    except (ImportError, FileNotFoundError) as e:
        logger.warning(f"Cannot create hand landmarker: {e}")
        cap.release()
        return observations

    try:
        # Process each timestamp
        for ts in timestamps:
            best_observation = None
            best_confidence = 0.0

            # Sample multiple frames around the onset
            for i in range(config.frames_per_onset):
                offset = (i - config.frames_per_onset // 2) * (
                    (config.frame_buffer_before + config.frame_buffer_after) /
                    config.frames_per_onset
                )
                sample_ts = ts + offset

                if sample_ts < 0:
                    continue

                frame_idx = int(sample_ts * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                # Apply ROI cropping if provided
                if roi is not None:
                    x1 = int(roi['x1'] * frame_width)
                    y1 = int(roi['y1'] * frame_height)
                    x2 = int(roi['x2'] * frame_width)
                    y2 = int(roi['y2'] * frame_height)
                    frame = frame[y1:y2, x1:x2]

                observation = detect_hand_landmarks(frame, config, landmarker)
                if observation is not None:
                    # Keep the observation with highest hand confidence
                    if observation.hand_confidence > best_confidence:
                        best_observation = observation
                        best_confidence = observation.hand_confidence

            if best_observation is not None:
                # Update timestamp to the original onset time
                best_observation = HandObservation(
                    timestamp=ts,
                    fingers=best_observation.fingers,
                    is_left_hand=best_observation.is_left_hand,
                    wrist_position=best_observation.wrist_position,
                    hand_confidence=best_observation.hand_confidence,
                    pressing_fingers=best_observation.pressing_fingers,
                )
                observations[ts] = best_observation
    finally:
        landmarker.close()
        cap.release()

    return observations
```

**Step 4: Run test to verify it passes**

Run: `cd tabvision-server && python -m pytest tests/test_video_pipeline.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tabvision-server/app/video_pipeline.py tabvision-server/tests/test_video_pipeline.py
git commit -m "$(cat <<'EOF'
feat(backend): pass ROI through video analysis pipeline

analyze_video_at_timestamps() now accepts optional roi parameter
to crop frames before hand detection.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Add ROI support to fretboard detection

**Files:**
- Modify: `tabvision-server/app/fretboard_detection.py:782-830`
- Test: `tabvision-server/tests/test_fretboard_detection.py`

**Step 1: Write the failing test**

Add to `tabvision-server/tests/test_fretboard_detection.py`:

```python
def test_detect_fretboard_from_video_with_roi(tmp_path):
    """detect_fretboard_from_video crops frames with ROI before detection."""
    import cv2

    video_path = str(tmp_path / "test.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 30, (200, 200))
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    for _ in range(30):
        out.write(frame)
    out.release()

    roi = {'x1': 0.25, 'y1': 0.25, 'x2': 0.75, 'y2': 0.75}

    # Track frame sizes passed to detect_fretboard
    captured_sizes = []

    def mock_detect(frame, config=None):
        captured_sizes.append(frame.shape[:2])
        return None  # Return None for simplicity

    with patch('app.fretboard_detection.detect_fretboard', side_effect=mock_detect):
        detect_fretboard_from_video(video_path, num_sample_frames=1, roi=roi)

    # Cropped frame should be 100x100 (half of 200x200)
    assert len(captured_sizes) > 0
    assert captured_sizes[0] == (100, 100)
```

**Step 2: Run test to verify it fails**

Run: `cd tabvision-server && python -m pytest tests/test_fretboard_detection.py::test_detect_fretboard_from_video_with_roi -v`
Expected: FAIL with "TypeError: detect_fretboard_from_video() got an unexpected keyword argument 'roi'"

**Step 3: Write minimal implementation**

Modify `tabvision-server/app/fretboard_detection.py` - update `detect_fretboard_from_video()`:

```python
def detect_fretboard_from_video(
    video_path: str,
    num_sample_frames: int = 5,
    roi: dict = None
) -> FretboardGeometry | None:
    """Detect fretboard geometry from video using multiple frames.

    Samples multiple frames and uses the detection with highest confidence.

    Args:
        video_path: Path to video file
        num_sample_frames: Number of frames to sample
        roi: Optional dict with x1, y1, x2, y2 (normalized 0-1) to crop frames

    Returns:
        FretboardGeometry with highest confidence, or None
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames <= 0:
        cap.release()
        return None

    # Sample frames evenly throughout the video (skip first/last 10%)
    start_frame = int(total_frames * 0.1)
    end_frame = int(total_frames * 0.9)
    frame_step = max(1, (end_frame - start_frame) // num_sample_frames)

    best_geometry = None
    best_confidence = 0.0

    for i in range(num_sample_frames):
        frame_idx = start_frame + i * frame_step
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # Apply ROI cropping if provided
        if roi is not None:
            x1 = int(roi['x1'] * frame_width)
            y1 = int(roi['y1'] * frame_height)
            x2 = int(roi['x2'] * frame_width)
            y2 = int(roi['y2'] * frame_height)
            frame = frame[y1:y2, x1:x2]

        geometry = detect_fretboard(frame)
        if geometry and geometry.detection_confidence > best_confidence:
            best_geometry = geometry
            best_confidence = geometry.detection_confidence

    cap.release()
    return best_geometry
```

**Step 4: Run test to verify it passes**

Run: `cd tabvision-server && python -m pytest tests/test_fretboard_detection.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tabvision-server/app/fretboard_detection.py tabvision-server/tests/test_fretboard_detection.py
git commit -m "$(cat <<'EOF'
feat(backend): add ROI support to fretboard detection

detect_fretboard_from_video() now accepts optional roi parameter
to crop frames before fretboard detection.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Pass ROI from Job through processing pipeline

**Files:**
- Modify: `tabvision-server/app/processing.py:129-283`
- Test: `tabvision-server/tests/test_processing.py`

**Step 1: Write the failing test**

Add to `tabvision-server/tests/test_processing.py`:

```python
def test_process_job_passes_roi_to_video_pipeline(tmp_path):
    """process_job passes ROI from job to video analysis functions."""
    from app.models import Job
    from app.storage import InMemoryJobStorage

    storage = InMemoryJobStorage()

    # Create job with ROI
    job = Job.create(video_path=str(tmp_path / "test.mp4"), capo_fret=0)
    job.roi_x1 = 0.1
    job.roi_y1 = 0.2
    job.roi_x2 = 0.8
    job.roi_y2 = 0.9
    storage.save(job)

    # Create minimal test video
    import cv2
    video_path = str(tmp_path / "test.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 30, (100, 100))
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    for _ in range(30):
        out.write(frame)
    out.release()
    job.video_path = video_path

    # Track ROI passed to functions
    captured_roi = {'fretboard': None, 'video': None}

    def mock_detect_fretboard(video_path, num_sample_frames=5, roi=None):
        captured_roi['fretboard'] = roi
        return None

    def mock_analyze_video(video_path, timestamps, config=None, roi=None):
        captured_roi['video'] = roi
        return {}

    with patch('app.processing.extract_audio'):
        with patch('app.processing.analyze_pitch', return_value=[]):
            with patch('app.processing.detect_note_onsets', return_value=[0.5]):
                with patch('app.processing.detect_fretboard_from_video', side_effect=mock_detect_fretboard):
                    with patch('app.processing.analyze_video_at_timestamps', side_effect=mock_analyze_video):
                        with patch('app.processing.fuse_audio_only', return_value=[]):
                            process_job(job.id, storage, str(tmp_path))

    expected_roi = {'x1': 0.1, 'y1': 0.2, 'x2': 0.8, 'y2': 0.9}
    assert captured_roi['fretboard'] == expected_roi
    assert captured_roi['video'] == expected_roi
```

**Step 2: Run test to verify it fails**

Run: `cd tabvision-server && python -m pytest tests/test_processing.py::test_process_job_passes_roi_to_video_pipeline -v`
Expected: FAIL (ROI not being passed)

**Step 3: Write minimal implementation**

Modify `tabvision-server/app/processing.py` - update `process_job()` to extract and pass ROI:

In the video analysis section (around line 180), update to:

```python
        # Build ROI dict from job if present
        roi = None
        if all(getattr(job, f'roi_{coord}', None) is not None
               for coord in ['x1', 'y1', 'x2', 'y2']):
            roi = {
                'x1': job.roi_x1,
                'y1': job.roi_y1,
                'x2': job.roi_x2,
                'y2': job.roi_y2,
            }

        try:
            # Get onset timestamps from detected notes
            timestamps = detect_note_onsets(detected_notes)
            logger.info(f"Detected {len(timestamps)} note onsets for video analysis")

            if timestamps:
                # Detect fretboard geometry using multiple frames for robustness
                fretboard = detect_fretboard_from_video(
                    job.video_path,
                    num_sample_frames=5,
                    roi=roi
                )

                if fretboard:
                    logger.info(
                        f"Fretboard detected with confidence {fretboard.detection_confidence:.2f}, "
                        f"{len(fretboard.fret_positions)} frets"
                    )

                    # For longer videos, track fretboard across time
                    if len(timestamps) > 10:
                        fretboard_timeline = track_fretboard_temporal(
                            job.video_path,
                            timestamps[::5],  # Sample every 5th onset
                            fretboard
                        )
                        # Use best fretboard from timeline if available
                        if fretboard_timeline:
                            best_fb = max(
                                fretboard_timeline.values(),
                                key=lambda fb: fb.detection_confidence
                            )
                            if best_fb.detection_confidence > fretboard.detection_confidence:
                                fretboard = best_fb
                                logger.info(
                                    f"Updated fretboard from temporal tracking: "
                                    f"confidence {fretboard.detection_confidence:.2f}"
                                )

                    # Analyze hand positions at note onset times
                    video_observations = analyze_video_at_timestamps(
                        job.video_path, timestamps, video_config, roi=roi
                    )
                    logger.info(
                        f"Video analysis: {len(video_observations)} hand observations "
                        f"from {len(timestamps)} onsets "
                        f"({len(video_observations)/len(timestamps)*100:.0f}% detection rate)"
                    )
                else:
                    logger.info("No fretboard detected, using audio-only mode")
```

**Step 4: Run test to verify it passes**

Run: `cd tabvision-server && python -m pytest tests/test_processing.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tabvision-server/app/processing.py tabvision-server/tests/test_processing.py
git commit -m "$(cat <<'EOF'
feat(backend): pass ROI from Job through processing pipeline

process_job() extracts ROI coordinates from Job and passes them
to detect_fretboard_from_video() and analyze_video_at_timestamps().

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Add roi_selection status to frontend store

**Files:**
- Modify: `tabvision-client/src/store/appStore.ts`

**Step 1: Update JobStatus type and add ROI state**

Modify `tabvision-client/src/store/appStore.ts`:

```typescript
// tabvision-client/src/store/appStore.ts
import { create } from 'zustand';
import { TabDocument, TabNote } from '../types/tab';

type JobStatus = 'idle' | 'roi_selection' | 'uploading' | 'processing' | 'completed' | 'failed';

interface ROI {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

interface EditAction {
  noteId: string;
  previousFret: number | "X";
  newFret: number | "X";
}

interface AppState {
  // Job state
  currentJobId: string | null;
  jobStatus: JobStatus;
  progress: number;
  currentStage: string;
  tabDocument: TabDocument | null;
  errorMessage: string | null;
  videoUrl: string | null;
  videoFile: File | null;  // Store the actual file for later upload
  roi: ROI | null;

  // Playback state
  currentTime: number;
  duration: number;
  isPlaying: boolean;

  // Editor state
  selectedNoteId: string | null;
  isFollowingPlayback: boolean;
  pendingFretInput: string;

  // Edit history
  editHistory: EditAction[];
  editHistoryIndex: number;

  // Job actions
  setJobId: (id: string) => void;
  setStatus: (status: JobStatus) => void;
  setProgress: (progress: number, stage: string) => void;
  setTabDocument: (doc: TabDocument) => void;
  setError: (message: string) => void;
  setVideoUrl: (url: string | null) => void;
  setVideoFile: (file: File | null) => void;
  setROI: (roi: ROI | null) => void;
  reset: () => void;

  // ... rest of actions unchanged
}

const initialState = {
  // Job state
  currentJobId: null,
  jobStatus: 'idle' as JobStatus,
  progress: 0,
  currentStage: '',
  tabDocument: null,
  errorMessage: null,
  videoUrl: null,
  videoFile: null,
  roi: null,

  // ... rest unchanged
};

export const useAppStore = create<AppState>((set, get) => ({
  ...initialState,

  // Job actions
  setJobId: (id) => set({ currentJobId: id }),
  setStatus: (status) => set({ jobStatus: status }),
  setProgress: (progress, stage) => set({ progress, currentStage: stage }),
  setTabDocument: (doc) => set({ tabDocument: doc, jobStatus: 'completed' }),
  setError: (message) => set({ errorMessage: message, jobStatus: 'failed' }),
  setVideoUrl: (url) => set({ videoUrl: url }),
  setVideoFile: (file) => set({ videoFile: file }),
  setROI: (roi) => set({ roi }),
  reset: () => set(initialState),

  // ... rest of implementation unchanged
}));
```

**Step 2: Commit**

```bash
git add tabvision-client/src/store/appStore.ts
git commit -m "$(cat <<'EOF'
feat(frontend): add ROI state and roi_selection status to store

Add roi field, videoFile field, and roi_selection to JobStatus type.
Add setROI and setVideoFile actions.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Add ROI parameters to API client

**Files:**
- Modify: `tabvision-client/src/api/client.ts`

**Step 1: Update uploadVideo function**

Modify `tabvision-client/src/api/client.ts`:

```typescript
// tabvision-client/src/api/client.ts
import { TabDocument, JobStatus } from '../types/tab';

const API_BASE = 'http://localhost:5000';

interface ROI {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export async function uploadVideo(
  file: File,
  capoFret: number = 0,
  roi?: ROI
): Promise<string> {
  const formData = new FormData();
  formData.append('video', file);
  formData.append('capo_fret', capoFret.toString());

  if (roi) {
    formData.append('roi_x1', roi.x1.toString());
    formData.append('roi_y1', roi.y1.toString());
    formData.append('roi_x2', roi.x2.toString());
    formData.append('roi_y2', roi.y2.toString());
  }

  const response = await fetch(`${API_BASE}/jobs`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Upload failed');
  }

  const data = await response.json();
  return data.job_id;
}

// ... rest unchanged
```

**Step 2: Commit**

```bash
git add tabvision-client/src/api/client.ts
git commit -m "$(cat <<'EOF'
feat(frontend): add ROI parameters to uploadVideo API call

uploadVideo() now accepts optional ROI object and appends
coordinates to FormData.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Create ROISelector component

**Files:**
- Create: `tabvision-client/src/components/ROISelector.tsx`

**Step 1: Create the component**

Create `tabvision-client/src/components/ROISelector.tsx`:

```typescript
// tabvision-client/src/components/ROISelector.tsx
import React, { useRef, useState, useEffect, useCallback } from 'react';
import { useAppStore } from '../store/appStore';
import { uploadVideo, getJobStatus, getJobResult } from '../api/client';

const MIN_BOX_SIZE = 0.1; // Minimum 10% of video dimensions

interface DrawState {
  isDrawing: boolean;
  startX: number;
  startY: number;
}

export function ROISelector() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const [drawState, setDrawState] = useState<DrawState>({
    isDrawing: false,
    startX: 0,
    startY: 0,
  });
  const [showHint, setShowHint] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const {
    videoUrl,
    videoFile,
    roi,
    setROI,
    setStatus,
    setJobId,
    setProgress,
    setTabDocument,
    setError: setAppError,
    reset,
  } = useAppStore();

  // Draw the ROI box on canvas
  const drawROI = useCallback(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (roi) {
      const x = roi.x1 * canvas.width;
      const y = roi.y1 * canvas.height;
      const width = (roi.x2 - roi.x1) * canvas.width;
      const height = (roi.y2 - roi.y1) * canvas.height;

      // Draw semi-transparent fill
      ctx.fillStyle = 'rgba(255, 255, 255, 0.1)';
      ctx.fillRect(x, y, width, height);

      // Draw border
      ctx.strokeStyle = '#00FFFF';
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, width, height);
    }
  }, [roi]);

  // Sync canvas size with video
  useEffect(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    const updateCanvasSize = () => {
      canvas.width = video.clientWidth;
      canvas.height = video.clientHeight;
      drawROI();
    };

    video.addEventListener('loadedmetadata', updateCanvasSize);
    window.addEventListener('resize', updateCanvasSize);

    // Initial size
    if (video.readyState >= 1) {
      updateCanvasSize();
    }

    return () => {
      video.removeEventListener('loadedmetadata', updateCanvasSize);
      window.removeEventListener('resize', updateCanvasSize);
    };
  }, [drawROI]);

  // Redraw when ROI changes
  useEffect(() => {
    drawROI();
  }, [roi, drawROI]);

  const getCanvasCoords = (e: React.MouseEvent): { x: number; y: number } => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };

    const rect = canvas.getBoundingClientRect();
    return {
      x: (e.clientX - rect.left) / canvas.width,
      y: (e.clientY - rect.top) / canvas.height,
    };
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    const coords = getCanvasCoords(e);
    setDrawState({
      isDrawing: true,
      startX: coords.x,
      startY: coords.y,
    });
    setError(null);
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!drawState.isDrawing) return;

    const coords = getCanvasCoords(e);
    const x1 = Math.min(drawState.startX, coords.x);
    const y1 = Math.min(drawState.startY, coords.y);
    const x2 = Math.max(drawState.startX, coords.x);
    const y2 = Math.max(drawState.startY, coords.y);

    setROI({ x1, y1, x2, y2 });
  };

  const handleMouseUp = () => {
    if (!drawState.isDrawing) return;

    setDrawState({ isDrawing: false, startX: 0, startY: 0 });

    // Validate minimum size
    if (roi) {
      const width = roi.x2 - roi.x1;
      const height = roi.y2 - roi.y1;
      if (width < MIN_BOX_SIZE || height < MIN_BOX_SIZE) {
        setError('Box too small. Please draw a larger region.');
        setROI(null);
        return;
      }
      setShowHint(false);
    }
  };

  const handleBack = () => {
    reset();
  };

  const handleProcess = async () => {
    if (!videoFile || !roi) return;

    setStatus('uploading');

    try {
      const jobId = await uploadVideo(videoFile, 0, roi);
      setJobId(jobId);
      setStatus('processing');

      // Poll for status
      const pollInterval = setInterval(async () => {
        try {
          const status = await getJobStatus(jobId);
          setProgress(status.progress, status.current_stage);

          if (status.status === 'completed') {
            clearInterval(pollInterval);
            const result = await getJobResult(jobId);
            setTabDocument(result);
          } else if (status.status === 'failed') {
            clearInterval(pollInterval);
            setAppError(status.error_message || 'Processing failed');
          }
        } catch (err) {
          clearInterval(pollInterval);
          setAppError(err instanceof Error ? err.message : 'Unknown error');
        }
      }, 1000);
    } catch (err) {
      setAppError(err instanceof Error ? err.message : 'Upload failed');
    }
  };

  const canProcess = roi !== null;

  return (
    <div className="space-y-4">
      {/* Video with canvas overlay */}
      <div ref={containerRef} className="relative">
        <video
          ref={videoRef}
          src={videoUrl || undefined}
          className="w-full rounded-lg"
          controls
        />
        <canvas
          ref={canvasRef}
          className="absolute top-0 left-0 w-full h-full cursor-crosshair"
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
        />
      </div>

      {/* Hint text */}
      {showHint && (
        <p className="text-center text-gray-400 text-sm">
          Draw a box around the fretboard and left hand
        </p>
      )}

      {/* Error message */}
      {error && (
        <p className="text-center text-red-500 text-sm">{error}</p>
      )}

      {/* Buttons */}
      <div className="flex justify-between">
        <button
          onClick={handleBack}
          className="px-4 py-2 bg-gray-600 hover:bg-gray-700 rounded transition-colors"
        >
          ← Back
        </button>
        <button
          onClick={handleProcess}
          disabled={!canProcess}
          className={`px-4 py-2 rounded transition-colors ${
            canProcess
              ? 'bg-blue-600 hover:bg-blue-700'
              : 'bg-gray-700 text-gray-500 cursor-not-allowed'
          }`}
        >
          Process →
        </button>
      </div>
    </div>
  );
}
```

**Step 2: Commit**

```bash
git add tabvision-client/src/components/ROISelector.tsx
git commit -m "$(cat <<'EOF'
feat(frontend): create ROISelector component

Canvas overlay on video for drawing bounding box. Includes:
- Click-drag to draw rectangle
- Minimum size validation (10%)
- Cyan border with semi-transparent fill
- Hint text that fades after drawing
- Back and Process buttons

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: Update UploadPanel to transition to ROI selection

**Files:**
- Modify: `tabvision-client/src/components/UploadPanel.tsx`

**Step 1: Update UploadPanel**

Modify `tabvision-client/src/components/UploadPanel.tsx`:

```typescript
// tabvision-client/src/components/UploadPanel.tsx
import React, { useCallback, useState } from 'react';
import { useAppStore } from '../store/appStore';

const ALLOWED_TYPES = ['video/mp4', 'video/quicktime'];

export function UploadPanel() {
  const [isDragging, setIsDragging] = useState(false);
  const {
    jobStatus,
    progress,
    currentStage,
    errorMessage,
    setStatus,
    setError,
    setVideoUrl,
    setVideoFile,
    reset,
  } = useAppStore();

  const processFile = useCallback((file: File) => {
    if (!ALLOWED_TYPES.includes(file.type)) {
      setError('Please upload an MP4 or MOV file');
      return;
    }

    // Create a blob URL for the video player
    const videoUrl = URL.createObjectURL(file);
    setVideoUrl(videoUrl);
    setVideoFile(file);

    // Transition to ROI selection
    setStatus('roi_selection');
  }, [setStatus, setError, setVideoUrl, setVideoFile]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const file = e.dataTransfer.files[0];
    if (file) {
      processFile(file);
    }
  }, [processFile]);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      processFile(file);
    }
  }, [processFile]);

  const isProcessing = jobStatus === 'uploading' || jobStatus === 'processing';

  return (
    <div
      className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
        isDragging ? 'border-blue-500 bg-blue-500/10' : 'border-gray-600 hover:border-gray-500'
      } ${isProcessing ? 'opacity-50 pointer-events-none' : ''}`}
      onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
    >
      <div className="space-y-4">
        <div className="text-4xl">🎸</div>

        {jobStatus === 'idle' && (
          <>
            <p className="text-lg">Drop a video file here or click to upload</p>
            <p className="text-sm text-gray-400">Supports MP4 and MOV files</p>
            <label className="inline-block px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded cursor-pointer transition-colors">
              Choose File
              <input
                type="file"
                accept="video/mp4,video/quicktime"
                onChange={handleFileSelect}
                className="hidden"
              />
            </label>
          </>
        )}

        {jobStatus === 'uploading' && (
          <p className="text-lg">Uploading video...</p>
        )}

        {jobStatus === 'processing' && (
          <div className="space-y-2">
            <p className="text-lg">Processing: {currentStage}</p>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div
                className="bg-blue-600 h-2 rounded-full transition-all"
                style={{ width: `${progress * 100}%` }}
              />
            </div>
            <p className="text-sm text-gray-400">{Math.round(progress * 100)}%</p>
          </div>
        )}

        {jobStatus === 'completed' && (
          <div className="space-y-2">
            <p className="text-lg text-green-500">Processing complete!</p>
            <button
              onClick={reset}
              className="px-4 py-2 bg-gray-600 hover:bg-gray-700 rounded transition-colors"
            >
              Upload Another
            </button>
          </div>
        )}

        {jobStatus === 'failed' && (
          <div className="space-y-2">
            <p className="text-lg text-red-500">Error: {errorMessage}</p>
            <button
              onClick={reset}
              className="px-4 py-2 bg-gray-600 hover:bg-gray-700 rounded transition-colors"
            >
              Try Again
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
```

**Step 2: Commit**

```bash
git add tabvision-client/src/components/UploadPanel.tsx
git commit -m "$(cat <<'EOF'
feat(frontend): update UploadPanel to transition to ROI selection

After file selection, UploadPanel now:
- Stores file in state (instead of immediately uploading)
- Creates blob URL for preview
- Transitions to roi_selection status

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: Update App.tsx to render ROISelector

**Files:**
- Modify: `tabvision-client/src/App.tsx`

**Step 1: Update App.tsx**

Modify `tabvision-client/src/App.tsx`:

```typescript
// tabvision-client/src/App.tsx
import React, { useRef } from 'react';
import { UploadPanel } from './components/UploadPanel';
import { ROISelector } from './components/ROISelector';
import { VideoPlayer } from './components/VideoPlayer';
import { TabCanvas } from './components/TabCanvas';
import { TabToolbar } from './components/TabToolbar';
import { useAppStore } from './store/appStore';
import './index.css';

function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const { jobStatus, videoUrl } = useAppStore();

  const showEditor = jobStatus === 'completed';
  const showROISelector = jobStatus === 'roi_selection';
  const showUploadOrProcessing = jobStatus === 'idle' || jobStatus === 'uploading' || jobStatus === 'processing' || jobStatus === 'failed';

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100">
      <header className="border-b border-gray-800 px-6 py-4">
        <h1 className="text-2xl font-bold">TabVision</h1>
        <p className="text-sm text-gray-400">Automatic Guitar Tab Transcription</p>
      </header>

      <main className="container mx-auto px-6 py-8 space-y-6 max-w-6xl">
        {/* Upload panel - shown when idle, uploading, processing, or failed */}
        {showUploadOrProcessing && <UploadPanel />}

        {/* ROI selector - shown after upload, before processing */}
        {showROISelector && <ROISelector />}

        {/* Video player - shown when video is loaded and editing */}
        {videoUrl && (
          <div className={showEditor ? '' : 'hidden'}>
            <VideoPlayer videoRef={videoRef} />
          </div>
        )}

        {/* Toolbar - shown when editing */}
        {showEditor && <TabToolbar />}

        {/* Tab canvas - shown when editing */}
        {showEditor && <TabCanvas videoRef={videoRef} />}
      </main>

      <footer className="border-t border-gray-800 px-6 py-4 text-center text-sm text-gray-500">
        Phase 4: Editor UI
      </footer>
    </div>
  );
}

export default App;
```

**Step 2: Commit**

```bash
git add tabvision-client/src/App.tsx
git commit -m "$(cat <<'EOF'
feat(frontend): render ROISelector in App when roi_selection status

App.tsx now shows ROISelector component between upload and processing.
Flow: idle → roi_selection → uploading → processing → completed

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 12: Manual integration test

**Step 1: Start backend server**

```bash
cd tabvision-server && python run.py
```

**Step 2: Start frontend dev server**

```bash
cd tabvision-client && npm run dev
```

**Step 3: Test the flow**

1. Open http://localhost:5173
2. Upload a video file
3. Verify ROI selector appears with video preview
4. Draw a bounding box on the video
5. Verify "Process" button becomes enabled
6. Click "Process" and verify processing starts with ROI
7. Verify results appear in tab editor

**Step 4: Final commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
feat: complete ROI selection feature

Full implementation of user-defined region of interest selection:
- Frontend: ROISelector component with video preview and canvas overlay
- Backend: ROI coordinates stored in Job and used to crop frames
- Processing: Fretboard detection and hand tracking use cropped frames

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```
