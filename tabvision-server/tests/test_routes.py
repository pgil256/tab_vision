"""Tests for API routes."""
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
