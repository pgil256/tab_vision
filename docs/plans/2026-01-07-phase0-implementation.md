# Phase 0: Skeleton Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build end-to-end skeleton where uploading a video returns a fake TabDocument displayed in the UI.

**Architecture:** Electron + React frontend communicates with Flask backend over HTTP. Backend stores jobs in-memory and returns hardcoded fake TabDocument. Frontend polls for job status and renders result.

**Tech Stack:** Electron Forge + Vite + React 18 + TypeScript (frontend), Flask + Python (backend), Zustand (state)

---

## Part A: Backend (Flask)

### Task 1: Initialize Flask Project

**Files:**
- Create: `tabvision-server/app/__init__.py`
- Create: `tabvision-server/app/models.py`
- Create: `tabvision-server/app/storage.py`
- Create: `tabvision-server/app/routes.py`
- Create: `tabvision-server/run.py`
- Create: `tabvision-server/requirements.txt`
- Create: `tabvision-server/.gitignore`

**Step 1: Create directory structure**

```bash
mkdir -p tabvision-server/app
mkdir -p tabvision-server/uploads
```

**Step 2: Create requirements.txt**

```
flask==3.0.0
flask-cors==4.0.0
python-dotenv==1.0.0
```

**Step 3: Create .gitignore**

```
__pycache__/
*.py[cod]
.env
venv/
uploads/*
!uploads/.gitkeep
```

**Step 4: Create uploads/.gitkeep**

Empty file to preserve the uploads directory in git.

**Step 5: Commit**

```bash
git add tabvision-server/
git commit -m "chore: initialize flask project structure"
```

---

### Task 2: Implement Job Model

**Files:**
- Create: `tabvision-server/app/models.py`
- Create: `tabvision-server/tests/__init__.py`
- Create: `tabvision-server/tests/test_models.py`

**Step 1: Write the failing test**

```python
# tabvision-server/tests/test_models.py
from app.models import Job

def test_job_creation():
    job = Job.create(video_path="/uploads/test.mp4", capo_fret=0)

    assert job.id is not None
    assert job.status == "pending"
    assert job.video_path == "/uploads/test.mp4"
    assert job.capo_fret == 0
    assert job.progress == 0.0
    assert job.current_stage == "uploading"
    assert job.result_path is None
    assert job.error_message is None

def test_job_to_dict():
    job = Job.create(video_path="/uploads/test.mp4", capo_fret=2)
    data = job.to_dict()

    assert data["id"] == job.id
    assert data["status"] == "pending"
    assert data["progress"] == 0.0
    assert data["current_stage"] == "uploading"
```

**Step 2: Run test to verify it fails**

```bash
cd tabvision-server
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install pytest
pytest tests/test_models.py -v
```

Expected: FAIL with "cannot import name 'Job'"

**Step 3: Write minimal implementation**

```python
# tabvision-server/app/models.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import uuid

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

    @classmethod
    def create(cls, video_path: str, capo_fret: int) -> "Job":
        now = datetime.utcnow()
        return cls(
            id=str(uuid.uuid4()),
            status="pending",
            created_at=now,
            updated_at=now,
            video_path=video_path,
            capo_fret=capo_fret,
            progress=0.0,
            current_stage="uploading",
            result_path=None,
            error_message=None,
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "progress": self.progress,
            "current_stage": self.current_stage,
            "error_message": self.error_message,
        }
```

**Step 4: Create tests/__init__.py**

Empty file.

**Step 5: Run test to verify it passes**

```bash
pytest tests/test_models.py -v
```

Expected: PASS

**Step 6: Commit**

```bash
git add tabvision-server/
git commit -m "feat(backend): add Job model with create and to_dict"
```

---

### Task 3: Implement Job Storage

**Files:**
- Create: `tabvision-server/app/storage.py`
- Create: `tabvision-server/tests/test_storage.py`

**Step 1: Write the failing test**

```python
# tabvision-server/tests/test_storage.py
from app.storage import JobStorage
from app.models import Job

def test_storage_save_and_get():
    storage = JobStorage()
    job = Job.create(video_path="/uploads/test.mp4", capo_fret=0)

    storage.save(job)
    retrieved = storage.get(job.id)

    assert retrieved is not None
    assert retrieved.id == job.id

def test_storage_get_nonexistent():
    storage = JobStorage()
    retrieved = storage.get("nonexistent-id")

    assert retrieved is None
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_storage.py -v
```

Expected: FAIL with "cannot import name 'JobStorage'"

**Step 3: Write minimal implementation**

```python
# tabvision-server/app/storage.py
from typing import Dict, Optional
from app.models import Job

class JobStorage:
    def __init__(self):
        self._jobs: Dict[str, Job] = {}

    def save(self, job: Job) -> None:
        self._jobs[job.id] = job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

# Global instance for the application
job_storage = JobStorage()
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_storage.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add tabvision-server/
git commit -m "feat(backend): add in-memory JobStorage"
```

---

### Task 4: Implement Flask App and Routes

**Files:**
- Create: `tabvision-server/app/__init__.py`
- Create: `tabvision-server/app/routes.py`
- Create: `tabvision-server/app/fake_data.py`
- Create: `tabvision-server/run.py`
- Create: `tabvision-server/tests/test_routes.py`

**Step 1: Write the failing tests**

```python
# tabvision-server/tests/test_routes.py
import pytest
import io
from app import create_app

@pytest.fixture
def client():
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_post_jobs_creates_job(client):
    data = {
        'video': (io.BytesIO(b'fake video content'), 'test.mp4'),
        'capo_fret': '0'
    }
    response = client.post('/jobs', data=data, content_type='multipart/form-data')

    assert response.status_code == 201
    json_data = response.get_json()
    assert 'job_id' in json_data

def test_get_job_status(client):
    # First create a job
    data = {
        'video': (io.BytesIO(b'fake video content'), 'test.mp4'),
        'capo_fret': '0'
    }
    create_response = client.post('/jobs', data=data, content_type='multipart/form-data')
    job_id = create_response.get_json()['job_id']

    # Then get its status
    response = client.get(f'/jobs/{job_id}')

    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data['status'] == 'completed'
    assert json_data['progress'] == 1.0

def test_get_job_not_found(client):
    response = client.get('/jobs/nonexistent-id')
    assert response.status_code == 404

def test_get_job_result(client):
    # First create a job
    data = {
        'video': (io.BytesIO(b'fake video content'), 'test.mp4'),
        'capo_fret': '2'
    }
    create_response = client.post('/jobs', data=data, content_type='multipart/form-data')
    job_id = create_response.get_json()['job_id']

    # Then get the result
    response = client.get(f'/jobs/{job_id}/result')

    assert response.status_code == 200
    json_data = response.get_json()
    assert 'id' in json_data
    assert 'notes' in json_data
    assert json_data['capoFret'] == 2
    assert len(json_data['notes']) > 0

def test_get_result_not_found(client):
    response = client.get('/jobs/nonexistent-id/result')
    assert response.status_code == 404
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_routes.py -v
```

Expected: FAIL with "cannot import name 'create_app'"

**Step 3: Create fake_data.py**

```python
# tabvision-server/app/fake_data.py
import uuid
from datetime import datetime

def generate_fake_tab_document(job_id: str, capo_fret: int) -> dict:
    """Generate a fake TabDocument for testing the skeleton."""
    return {
        "id": job_id,
        "createdAt": datetime.utcnow().isoformat() + "Z",
        "duration": 30.0,
        "capoFret": capo_fret,
        "tuning": ["E", "A", "D", "G", "B", "E"],
        "notes": [
            {
                "id": str(uuid.uuid4()),
                "timestamp": 0.5,
                "string": 1,
                "fret": 0,
                "confidence": 0.95,
                "confidenceLevel": "high",
                "isEdited": False,
            },
            {
                "id": str(uuid.uuid4()),
                "timestamp": 1.0,
                "string": 2,
                "fret": 1,
                "confidence": 0.72,
                "confidenceLevel": "medium",
                "isEdited": False,
            },
            {
                "id": str(uuid.uuid4()),
                "timestamp": 1.5,
                "string": 3,
                "fret": 0,
                "confidence": 0.88,
                "confidenceLevel": "high",
                "isEdited": False,
            },
            {
                "id": str(uuid.uuid4()),
                "timestamp": 2.0,
                "string": 4,
                "fret": 2,
                "confidence": 0.45,
                "confidenceLevel": "low",
                "isEdited": False,
            },
            {
                "id": str(uuid.uuid4()),
                "timestamp": 2.5,
                "string": 5,
                "fret": 3,
                "confidence": 0.91,
                "confidenceLevel": "high",
                "isEdited": False,
            },
            {
                "id": str(uuid.uuid4()),
                "timestamp": 3.0,
                "string": 6,
                "fret": 3,
                "confidence": 0.67,
                "confidenceLevel": "medium",
                "isEdited": False,
            },
        ],
    }
```

**Step 4: Create routes.py**

```python
# tabvision-server/app/routes.py
import os
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from app.models import Job
from app.storage import job_storage
from app.fake_data import generate_fake_tab_document

bp = Blueprint('jobs', __name__)

ALLOWED_EXTENSIONS = {'mp4', 'mov'}

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

    # Save the file
    filename = secure_filename(file.filename)
    upload_folder = current_app.config['UPLOAD_FOLDER']
    os.makedirs(upload_folder, exist_ok=True)

    # Create job first to get ID for unique filename
    job = Job.create(video_path="", capo_fret=capo_fret)

    # Use job ID in filename to ensure uniqueness
    ext = filename.rsplit('.', 1)[1].lower()
    unique_filename = f"{job.id}.{ext}"
    file_path = os.path.join(upload_folder, unique_filename)
    file.save(file_path)

    # Update job with file path and mark as completed (fake processing)
    job.video_path = file_path
    job.status = "completed"
    job.progress = 1.0
    job.current_stage = "complete"
    job.result_path = f"/results/{job.id}.json"

    job_storage.save(job)

    return jsonify({'job_id': job.id}), 201

@bp.route('/jobs/<job_id>', methods=['GET'])
def get_job_status(job_id: str):
    job = job_storage.get(job_id)
    if job is None:
        return jsonify({'error': 'Job not found'}), 404

    return jsonify(job.to_dict()), 200

@bp.route('/jobs/<job_id>/result', methods=['GET'])
def get_job_result(job_id: str):
    job = job_storage.get(job_id)
    if job is None:
        return jsonify({'error': 'Job not found'}), 404

    if job.status != "completed":
        return jsonify({'error': 'Job not completed yet'}), 400

    tab_document = generate_fake_tab_document(job.id, job.capo_fret)
    return jsonify(tab_document), 200
```

**Step 5: Create app/__init__.py**

```python
# tabvision-server/app/__init__.py
import os
from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__)

    # Configuration
    app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
    app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload

    # Enable CORS for local development
    CORS(app, origins=['http://localhost:*', 'http://127.0.0.1:*'])

    # Register blueprints
    from app.routes import bp
    app.register_blueprint(bp)

    return app
```

**Step 6: Create run.py**

```python
# tabvision-server/run.py
from app import create_app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

**Step 7: Run tests to verify they pass**

```bash
pytest tests/ -v
```

Expected: All tests PASS

**Step 8: Manual test**

```bash
python run.py
# In another terminal:
curl -X POST -F "video=@/path/to/any/video.mp4" -F "capo_fret=0" http://localhost:5000/jobs
```

**Step 9: Commit**

```bash
git add tabvision-server/
git commit -m "feat(backend): implement /jobs endpoints with fake processing"
```

---

## Part B: Frontend (Electron + React)

### Task 5: Initialize Electron Forge Project

**Files:**
- Create: `tabvision-client/` (entire project via Electron Forge CLI)

**Step 1: Create Electron Forge project with Vite + React + TypeScript**

```bash
cd /home/gilhooleyp/projects/tab_vision
npm create electron-app@latest tabvision-client -- --template=vite-typescript
```

**Step 2: Install additional dependencies**

```bash
cd tabvision-client
npm install zustand
npm install -D tailwindcss postcss autoprefixer @types/node
npx tailwindcss init -p
```

**Step 3: Verify it runs**

```bash
npm run dev
```

Expected: Electron window opens with default Vite template.

**Step 4: Commit**

```bash
git add tabvision-client/
git commit -m "chore: initialize electron forge with vite + typescript"
```

---

### Task 6: Configure Tailwind CSS

**Files:**
- Modify: `tabvision-client/tailwind.config.js`
- Modify: `tabvision-client/src/index.css`

**Step 1: Update tailwind.config.js**

```javascript
// tabvision-client/tailwind.config.js
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
```

**Step 2: Update src/index.css**

```css
/* tabvision-client/src/index.css */
@tailwind base;
@tailwind components;
@tailwind utilities;

body {
  @apply bg-gray-900 text-gray-100;
}
```

**Step 3: Verify Tailwind works**

Update App.tsx temporarily to include a Tailwind class like `className="text-blue-500"` and verify it applies.

```bash
npm run dev
```

**Step 4: Commit**

```bash
git add tabvision-client/
git commit -m "chore: configure tailwind css"
```

---

### Task 7: Create TypeScript Types

**Files:**
- Create: `tabvision-client/src/types/tab.ts`

**Step 1: Create the types file**

```typescript
// tabvision-client/src/types/tab.ts

export interface TabNote {
  id: string;
  timestamp: number;
  string: 1 | 2 | 3 | 4 | 5 | 6;
  fret: number | "X";
  confidence: number;
  confidenceLevel: "high" | "medium" | "low";
  isEdited: boolean;
  originalFret?: number | "X";
  detectedPitch?: number;
  detectedMidiNote?: number;
}

export interface TabDocument {
  id: string;
  createdAt: string;
  duration: number;
  capoFret: number;
  tuning: string[];
  notes: TabNote[];
}

export interface JobStatus {
  id: string;
  status: "pending" | "processing" | "completed" | "failed";
  progress: number;
  current_stage: string;
  error_message?: string;
}
```

**Step 2: Commit**

```bash
git add tabvision-client/
git commit -m "feat(frontend): add TabDocument and JobStatus types"
```

---

### Task 8: Create Zustand Store

**Files:**
- Create: `tabvision-client/src/store/appStore.ts`

**Step 1: Create the store**

```typescript
// tabvision-client/src/store/appStore.ts
import { create } from 'zustand';
import { TabDocument } from '../types/tab';

type JobStatus = 'idle' | 'uploading' | 'processing' | 'completed' | 'failed';

interface AppState {
  currentJobId: string | null;
  jobStatus: JobStatus;
  progress: number;
  currentStage: string;
  tabDocument: TabDocument | null;
  errorMessage: string | null;

  // Actions
  setJobId: (id: string) => void;
  setStatus: (status: JobStatus) => void;
  setProgress: (progress: number, stage: string) => void;
  setTabDocument: (doc: TabDocument) => void;
  setError: (message: string) => void;
  reset: () => void;
}

const initialState = {
  currentJobId: null,
  jobStatus: 'idle' as JobStatus,
  progress: 0,
  currentStage: '',
  tabDocument: null,
  errorMessage: null,
};

export const useAppStore = create<AppState>((set) => ({
  ...initialState,

  setJobId: (id) => set({ currentJobId: id }),

  setStatus: (status) => set({ jobStatus: status }),

  setProgress: (progress, stage) => set({ progress, currentStage: stage }),

  setTabDocument: (doc) => set({ tabDocument: doc, jobStatus: 'completed' }),

  setError: (message) => set({ errorMessage: message, jobStatus: 'failed' }),

  reset: () => set(initialState),
}));
```

**Step 2: Commit**

```bash
git add tabvision-client/
git commit -m "feat(frontend): add zustand store for app state"
```

---

### Task 9: Create API Client

**Files:**
- Create: `tabvision-client/src/api/client.ts`

**Step 1: Create the API client**

```typescript
// tabvision-client/src/api/client.ts
import { TabDocument, JobStatus } from '../types/tab';

const API_BASE = 'http://localhost:5000';

export async function uploadVideo(file: File, capoFret: number = 0): Promise<string> {
  const formData = new FormData();
  formData.append('video', file);
  formData.append('capo_fret', capoFret.toString());

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

export async function getJobStatus(jobId: string): Promise<JobStatus> {
  const response = await fetch(`${API_BASE}/jobs/${jobId}`);

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Failed to get job status');
  }

  const data = await response.json();
  return {
    id: jobId,
    status: data.status,
    progress: data.progress,
    current_stage: data.current_stage,
    error_message: data.error_message,
  };
}

export async function getJobResult(jobId: string): Promise<TabDocument> {
  const response = await fetch(`${API_BASE}/jobs/${jobId}/result`);

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Failed to get result');
  }

  return response.json();
}
```

**Step 2: Commit**

```bash
git add tabvision-client/
git commit -m "feat(frontend): add API client for backend communication"
```

---

### Task 10: Create Upload Component

**Files:**
- Create: `tabvision-client/src/components/UploadPanel.tsx`

**Step 1: Create the component**

```typescript
// tabvision-client/src/components/UploadPanel.tsx
import React, { useCallback, useState } from 'react';
import { useAppStore } from '../store/appStore';
import { uploadVideo, getJobStatus, getJobResult } from '../api/client';

const ALLOWED_TYPES = ['video/mp4', 'video/quicktime'];

export function UploadPanel() {
  const [isDragging, setIsDragging] = useState(false);
  const { jobStatus, progress, currentStage, errorMessage, setJobId, setStatus, setProgress, setTabDocument, setError, reset } = useAppStore();

  const processFile = useCallback(async (file: File) => {
    if (!ALLOWED_TYPES.includes(file.type)) {
      setError('Please upload an MP4 or MOV file');
      return;
    }

    reset();
    setStatus('uploading');

    try {
      const jobId = await uploadVideo(file, 0);
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
            setError(status.error_message || 'Processing failed');
          }
        } catch (err) {
          clearInterval(pollInterval);
          setError(err instanceof Error ? err.message : 'Unknown error');
        }
      }, 1000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    }
  }, [reset, setJobId, setStatus, setProgress, setTabDocument, setError]);

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
git add tabvision-client/
git commit -m "feat(frontend): add UploadPanel component with drag-drop"
```

---

### Task 11: Create Tab Editor Placeholder

**Files:**
- Create: `tabvision-client/src/components/TabEditor.tsx`

**Step 1: Create the component**

```typescript
// tabvision-client/src/components/TabEditor.tsx
import React from 'react';
import { useAppStore } from '../store/appStore';
import { TabNote } from '../types/tab';

const STRING_NAMES = ['e', 'B', 'G', 'D', 'A', 'E'];

function getConfidenceColor(level: TabNote['confidenceLevel']): string {
  switch (level) {
    case 'high': return 'text-green-500';
    case 'medium': return 'text-yellow-500';
    case 'low': return 'text-red-500';
    default: return 'text-gray-400';
  }
}

export function TabEditor() {
  const { tabDocument, jobStatus } = useAppStore();

  if (jobStatus !== 'completed' || !tabDocument) {
    return (
      <div className="border border-gray-700 rounded-lg p-8 text-center text-gray-500">
        <p>Upload a video to see the tab here</p>
      </div>
    );
  }

  // Group notes by approximate timestamp (for simple display)
  // In a real implementation, this would be more sophisticated
  const sortedNotes = [...tabDocument.notes].sort((a, b) => a.timestamp - b.timestamp);

  // Create a simple text-based tab display
  const renderSimpleTab = () => {
    // Initialize strings with dashes
    const strings: string[][] = STRING_NAMES.map(() => []);
    const colors: string[][] = STRING_NAMES.map(() => []);

    // Add notes at their positions
    sortedNotes.forEach((note) => {
      const stringIndex = note.string - 1;
      const fretStr = note.fret === 'X' ? 'X' : note.fret.toString();
      strings[stringIndex].push(fretStr.padStart(2, '-'));
      colors[stringIndex].push(getConfidenceColor(note.confidenceLevel));
    });

    return (
      <div className="font-mono text-sm space-y-1">
        {STRING_NAMES.map((name, idx) => (
          <div key={name} className="flex">
            <span className="w-4 text-gray-500">{name}|</span>
            <span className="flex">
              {strings[idx].length > 0 ? (
                strings[idx].map((fret, i) => (
                  <span key={i} className={`${colors[idx][i]} mx-1`}>
                    {fret}
                  </span>
                ))
              ) : (
                <span className="text-gray-600">---</span>
              )}
              <span className="text-gray-600">---|</span>
            </span>
          </div>
        ))}
      </div>
    );
  };

  return (
    <div className="border border-gray-700 rounded-lg p-6 space-y-4">
      <div className="flex justify-between items-center">
        <h2 className="text-lg font-semibold">Tab Output</h2>
        <div className="text-sm text-gray-400">
          {tabDocument.notes.length} notes detected
        </div>
      </div>

      <div className="bg-gray-800 rounded p-4 overflow-x-auto">
        {renderSimpleTab()}
      </div>

      <div className="flex gap-4 text-sm">
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 rounded-full bg-green-500"></span>
          High confidence
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 rounded-full bg-yellow-500"></span>
          Medium
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 rounded-full bg-red-500"></span>
          Low
        </span>
      </div>

      <div className="text-xs text-gray-500">
        Tuning: {tabDocument.tuning.join(' ')} | Capo: {tabDocument.capoFret || 'None'} | Duration: {tabDocument.duration}s
      </div>
    </div>
  );
}
```

**Step 2: Commit**

```bash
git add tabvision-client/
git commit -m "feat(frontend): add TabEditor placeholder component"
```

---

### Task 12: Wire Up App Component

**Files:**
- Modify: `tabvision-client/src/App.tsx`

**Step 1: Update App.tsx**

```typescript
// tabvision-client/src/App.tsx
import React from 'react';
import { UploadPanel } from './components/UploadPanel';
import { TabEditor } from './components/TabEditor';
import './index.css';

function App() {
  return (
    <div className="min-h-screen bg-gray-900 text-gray-100">
      <header className="border-b border-gray-800 px-6 py-4">
        <h1 className="text-2xl font-bold">TabVision</h1>
        <p className="text-sm text-gray-400">Automatic Guitar Tab Transcription</p>
      </header>

      <main className="container mx-auto px-6 py-8 space-y-8 max-w-4xl">
        <UploadPanel />
        <TabEditor />
      </main>

      <footer className="border-t border-gray-800 px-6 py-4 text-center text-sm text-gray-500">
        Phase 0: Skeleton
      </footer>
    </div>
  );
}

export default App;
```

**Step 2: Verify the full app works**

```bash
# Terminal 1: Start backend
cd tabvision-server
source venv/bin/activate
python run.py

# Terminal 2: Start frontend
cd tabvision-client
npm run dev
```

Expected: Electron app opens, can upload a video, see progress, and fake tab data displays.

**Step 3: Commit**

```bash
git add tabvision-client/
git commit -m "feat(frontend): wire up App with UploadPanel and TabEditor"
```

---

## Part C: Final Integration

### Task 13: End-to-End Test

**Step 1: Manual end-to-end test**

1. Start backend: `cd tabvision-server && python run.py`
2. Start frontend: `cd tabvision-client && npm run dev`
3. Upload any MP4 or MOV file
4. Verify:
   - Progress shows "Uploading..." then "Processing: complete"
   - Progress bar fills to 100%
   - Tab editor shows 6 fake notes with confidence colors
   - "Upload Another" button works

**Step 2: Update CLAUDE.md with working commands**

Verify the commands in CLAUDE.md match reality.

**Step 3: Final commit**

```bash
git add .
git commit -m "feat: complete phase 0 skeleton - end-to-end upload to fake tab"
```

---

## Summary

| Task | Description | Est. Time |
|------|-------------|-----------|
| 1 | Initialize Flask project | 5 min |
| 2 | Implement Job model | 10 min |
| 3 | Implement Job storage | 5 min |
| 4 | Implement Flask routes | 15 min |
| 5 | Initialize Electron Forge | 10 min |
| 6 | Configure Tailwind | 5 min |
| 7 | Create TypeScript types | 5 min |
| 8 | Create Zustand store | 5 min |
| 9 | Create API client | 5 min |
| 10 | Create UploadPanel | 15 min |
| 11 | Create TabEditor | 15 min |
| 12 | Wire up App | 5 min |
| 13 | End-to-end test | 10 min |

**Total: ~13 tasks, ~110 minutes**

---

## Deliverable

Upload a video → see fake TabDocument rendered in placeholder editor.
