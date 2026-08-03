"""Local-only gold-session endpoint: gating, guards, and the happy path."""
import io
import json
import os
from unittest.mock import patch

import pytest

from app import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv('TABVISION_PERSONAL_ROOT', str(tmp_path / 'personal'))
    app = create_app({'UPLOAD_FOLDER': str(tmp_path / 'uploads')})
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def _make_completed_job(client, tmp_path, *, capo='0', ext='webm'):
    with patch('app.routes.Thread'):
        response = client.post(
            '/jobs',
            data={'video': (io.BytesIO(b'fake media'), f'take.{ext}'), 'capo_fret': capo},
            content_type='multipart/form-data',
        )
    job_id = response.get_json()['job_id']
    from app.routes import get_job_storage
    storage = get_job_storage()
    job = storage.get(job_id)
    job.status = 'completed'
    storage.save(job)
    return job_id


NOTES = [
    {'timestamp': 1.5, 'string': 6, 'fret': 3},
    {'timestamp': 2.0, 'string': 1, 'fret': 0},
]


def test_endpoint_is_404_when_not_enabled(tmp_path, monkeypatch):
    monkeypatch.delenv('TABVISION_PERSONAL_ROOT', raising=False)
    app = create_app({'UPLOAD_FOLDER': str(tmp_path / 'uploads')})
    app.config['TESTING'] = True
    with app.test_client() as client:
        response = client.post('/jobs/whatever/gold-session', json={'notes': NOTES})

    assert response.status_code == 404
    assert 'not enabled' in response.get_json()['error']


def test_health_advertises_the_capability(client, tmp_path, monkeypatch):
    assert client.get('/health').get_json()['personal_ingest'] is True

    monkeypatch.delenv('TABVISION_PERSONAL_ROOT')
    assert client.get('/health').get_json()['personal_ingest'] is False


def test_guards_unknown_incomplete_capo_and_bad_body(client, tmp_path):
    assert client.post('/jobs/missing/gold-session', json={'notes': NOTES}).status_code == 404

    capo_job = _make_completed_job(client, tmp_path, capo='2')
    response = client.post(f'/jobs/{capo_job}/gold-session', json={'notes': NOTES})
    assert response.status_code == 400
    assert 'capo 0' in response.get_json()['error']

    job_id = _make_completed_job(client, tmp_path)
    assert client.post(f'/jobs/{job_id}/gold-session', json={}).status_code == 400
    bad_notes = client.post(
        f'/jobs/{job_id}/gold-session', json={'notes': [{'timestamp': 1, 'string': 9, 'fret': 0}]}
    )
    assert bad_notes.status_code == 400


def test_video_job_banks_frames_and_prior_labels(client, tmp_path):
    job_id = _make_completed_job(client, tmp_path, ext='webm')

    class _FakeDemux:
        frame_iterator = iter(())

    with patch('tabvision.demux.demux', return_value=_FakeDemux()) as demux_mock:
        response = client.post(f'/jobs/{job_id}/gold-session', json={'notes': NOTES})

    assert response.status_code == 200
    summary = response.get_json()
    assert summary['notes'] == 2
    assert summary['prior_labels'] == 2
    demux_mock.assert_called_once()

    root = os.environ['TABVISION_PERSONAL_ROOT']
    rows = (
        open(os.path.join(root, 'labels.jsonl'), encoding='utf-8').read().strip().splitlines()
    )
    assert len(rows) == 2
    parsed = [json.loads(row) for row in rows]
    assert {row['source'] for row in parsed} == {'studio-correction'}
    assert parsed[0]['string_idx'] == 0 and parsed[0]['fret'] == 3
    # The corpus session dir exists with its (empty-frame) manifest.
    session_dir = summary['session_dir']
    assert os.path.isfile(os.path.join(session_dir, 'meta.json'))


def test_audio_only_job_skips_frames_but_banks_labels(client, tmp_path):
    job_id = _make_completed_job(client, tmp_path, ext='wav')

    with patch('tabvision.demux.demux') as demux_mock:
        response = client.post(f'/jobs/{job_id}/gold-session', json={'notes': NOTES})

    assert response.status_code == 200
    summary = response.get_json()
    assert summary['frames_written'] == 0
    assert summary['session_dir'] is None
    assert summary['prior_labels'] == 2
    demux_mock.assert_not_called()
