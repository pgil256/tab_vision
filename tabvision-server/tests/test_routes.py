"""Tests for API routes."""
import pytest
import io
import json
import os
from unittest.mock import patch, MagicMock
from app import create_app
from app.storage import job_storage


@pytest.fixture
def client():
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_post_jobs_creates_job(client):
    with patch('app.routes.Thread') as mock_thread:
        data = {
            'video': (io.BytesIO(b'fake video content'), 'test.mp4'),
            'capo_fret': '0'
        }
        response = client.post('/jobs', data=data, content_type='multipart/form-data')

        assert response.status_code == 201
        json_data = response.get_json()
        assert 'job_id' in json_data
        # Verify thread was started
        mock_thread.return_value.start.assert_called_once()


def test_get_job_status(client):
    with patch('app.routes.Thread'):
        # First create a job
        data = {
            'video': (io.BytesIO(b'fake video content'), 'test.mp4'),
            'capo_fret': '0'
        }
        create_response = client.post('/jobs', data=data, content_type='multipart/form-data')
        job_id = create_response.get_json()['job_id']

        # Then get its status (will be pending since processing is mocked)
        response = client.get(f'/jobs/{job_id}')

        assert response.status_code == 200
        json_data = response.get_json()
        assert json_data['id'] == job_id
        # Status will be pending since we mocked the thread
        assert json_data['status'] in ['pending', 'processing', 'completed']


def test_get_job_not_found(client):
    response = client.get('/jobs/nonexistent-id')
    assert response.status_code == 404


def test_get_job_result(client, tmp_path):
    """Test getting result for a completed job."""
    with patch('app.routes.Thread'):
        # First create a job
        data = {
            'video': (io.BytesIO(b'fake video content'), 'test.mp4'),
            'capo_fret': '2'
        }
        create_response = client.post('/jobs', data=data, content_type='multipart/form-data')
        job_id = create_response.get_json()['job_id']

        # Manually complete the job with a result file
        job = job_storage.get(job_id)
        job.status = "completed"

        # Create result file
        result_data = {
            "notes": [
                {
                    "id": "test-1",
                    "timestamp": 1.0,
                    "string": 1,
                    "fret": 5,
                    "confidence": 0.9,
                    "confidenceLevel": "high"
                }
            ],
            "metadata": {
                "job_id": job_id,
                "capo_fret": 2
            }
        }
        result_path = str(tmp_path / f"{job_id}_result.json")
        with open(result_path, 'w') as f:
            json.dump(result_data, f)
        job.result_path = result_path

        # Then get the result
        response = client.get(f'/jobs/{job_id}/result')

        assert response.status_code == 200
        json_data = response.get_json()
        assert 'notes' in json_data
        assert len(json_data['notes']) == 1


def test_get_result_not_completed(client):
    """Test that getting result for incomplete job returns 400."""
    with patch('app.routes.Thread'):
        # Create a job (will be pending)
        data = {
            'video': (io.BytesIO(b'fake video content'), 'test.mp4'),
            'capo_fret': '0'
        }
        create_response = client.post('/jobs', data=data, content_type='multipart/form-data')
        job_id = create_response.get_json()['job_id']

        # Try to get result before completion
        response = client.get(f'/jobs/{job_id}/result')

        assert response.status_code == 400
        assert 'not completed' in response.get_json()['error']


def test_get_result_not_found(client):
    response = client.get('/jobs/nonexistent-id/result')
    assert response.status_code == 404


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
