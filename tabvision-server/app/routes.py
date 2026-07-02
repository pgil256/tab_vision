"""API routes for TabVision."""
import os
from threading import Thread
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from app.models import Job
from app.storage import job_storage
from app.result_io import load_result

bp = Blueprint('jobs', __name__)

ALLOWED_EXTENSIONS = {'mp4', 'mov', 'webm', 'wav', 'mp3', 'm4a'}
ALLOWED_INSTRUMENTS = {'acoustic', 'electric', 'classical'}
ALLOWED_TONES = {'clean', 'distorted'}
ALLOWED_STYLES = {'fingerstyle', 'strumming', 'mixed'}
ALLOWED_ACCURACY_MODES = {'fast', 'accurate'}


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_job_storage():
    return current_app.config.get('JOB_STORAGE', job_storage)


def dispatch_local_job(job_id: str, storage, results_folder: str) -> None:
    from app.processing import process_job

    thread = Thread(
        target=process_job,
        args=(job_id, storage, results_folder),
        daemon=True
    )
    thread.start()


def get_job_dispatcher():
    return current_app.config.get('JOB_DISPATCHER', dispatch_local_job)


def get_result_loader():
    return current_app.config.get('RESULT_LOADER', load_result)


def run_configured_hook(name: str) -> None:
    hook = current_app.config.get(name)
    if hook:
        hook()


def parse_choice(name: str, allowed: set[str], default: str):
    value = request.form.get(name, default).strip().lower()
    if value not in allowed:
        allowed_values = ', '.join(sorted(allowed))
        return None, jsonify({'error': f'{name} must be one of: {allowed_values}'}), 400
    return value, None, None


@bp.route('/jobs', methods=['POST'])
def create_job():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed. Use MP4, MOV, WEBM, WAV, MP3, or M4A.'}), 400

    capo_fret = request.form.get('capo_fret', '0')
    try:
        capo_fret = int(capo_fret)
    except ValueError:
        capo_fret = 0

    instrument, error_response, status_code = parse_choice(
        'instrument', ALLOWED_INSTRUMENTS, 'acoustic'
    )
    if error_response:
        return error_response, status_code
    tone, error_response, status_code = parse_choice('tone', ALLOWED_TONES, 'clean')
    if error_response:
        return error_response, status_code
    style, error_response, status_code = parse_choice('style', ALLOWED_STYLES, 'mixed')
    if error_response:
        return error_response, status_code
    accuracy_mode, error_response, status_code = parse_choice(
        'accuracy_mode', ALLOWED_ACCURACY_MODES, 'accurate'
    )
    if error_response:
        return error_response, status_code

    # Parse ROI coordinates if provided
    roi_x1 = request.form.get('roi_x1')
    roi_y1 = request.form.get('roi_y1')
    roi_x2 = request.form.get('roi_x2')
    roi_y2 = request.form.get('roi_y2')

    roi_values = None
    if any([roi_x1, roi_y1, roi_x2, roi_y2]):
        # If any ROI value is provided, all must be provided
        if not all([roi_x1, roi_y1, roi_x2, roi_y2]):
            return jsonify({'error': 'ROI requires all four coordinates: roi_x1, roi_y1, roi_x2, roi_y2'}), 400

        try:
            roi_values = {
                'x1': float(roi_x1),
                'y1': float(roi_y1),
                'x2': float(roi_x2),
                'y2': float(roi_y2),
            }
        except ValueError:
            return jsonify({'error': 'ROI coordinates must be valid numbers'}), 400

        # Validate range (0-1)
        for key, val in roi_values.items():
            if val < 0 or val > 1:
                return jsonify({'error': f'ROI coordinates must be in 0-1 range'}), 400

        # Validate order (x1 < x2, y1 < y2)
        if roi_values['x1'] >= roi_values['x2']:
            return jsonify({'error': 'ROI x1 must be less than x2'}), 400
        if roi_values['y1'] >= roi_values['y2']:
            return jsonify({'error': 'ROI y1 must be less than y2'}), 400

    # Save the file
    filename = secure_filename(file.filename)
    upload_folder = current_app.config['UPLOAD_FOLDER']
    os.makedirs(upload_folder, exist_ok=True)

    # Create job first to get ID for unique filename
    job = Job.create(
        video_path="",
        capo_fret=capo_fret,
        instrument=instrument,
        tone=tone,
        style=style,
        accuracy_mode=accuracy_mode,
    )

    # Use job ID in filename to ensure uniqueness
    ext = filename.rsplit('.', 1)[1].lower()
    unique_filename = f"{job.id}.{ext}"
    file_path = os.path.join(upload_folder, unique_filename)
    file.save(file_path)

    # Update job with file path and ROI
    job.video_path = file_path
    if roi_values:
        job.roi_x1 = roi_values['x1']
        job.roi_y1 = roi_values['y1']
        job.roi_x2 = roi_values['x2']
        job.roi_y2 = roi_values['y2']
    storage = get_job_storage()
    storage.save(job)
    run_configured_hook('UPLOAD_SAVED_HOOK')

    # Launch background processing
    results_folder = current_app.config.get('RESULTS_FOLDER', upload_folder)
    get_job_dispatcher()(job.id, storage, results_folder)

    return jsonify({'job_id': job.id}), 201


@bp.route('/jobs/<job_id>', methods=['GET'])
def get_job_status(job_id: str):
    job = get_job_storage().get(job_id)
    if job is None:
        return jsonify({'error': 'Job not found'}), 404

    return jsonify(job.to_dict()), 200


@bp.route('/jobs/<job_id>/result', methods=['GET'])
def get_job_result(job_id: str):
    job = get_job_storage().get(job_id)
    if job is None:
        return jsonify({'error': 'Job not found'}), 404

    if job.status != "completed":
        return jsonify({'error': 'Job not completed yet'}), 400

    try:
        run_configured_hook('RESULTS_RELOAD_HOOK')
        tab_document = get_result_loader()(job)
    except FileNotFoundError:
        return jsonify({'error': 'Result file not found'}), 500

    return jsonify(tab_document), 200
