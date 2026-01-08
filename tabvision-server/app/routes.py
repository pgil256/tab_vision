"""API routes for TabVision."""
import os
from threading import Thread
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from app.models import Job
from app.storage import job_storage
from app.processing import process_job, load_result

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

    try:
        tab_document = load_result(job)
    except FileNotFoundError:
        return jsonify({'error': 'Result file not found'}), 500

    return jsonify(tab_document), 200
