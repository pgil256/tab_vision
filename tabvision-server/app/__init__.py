"""Flask application factory."""
import os
import logging
from flask import Flask, jsonify
from flask_cors import CORS

logger = logging.getLogger(__name__)


def _prewarm_ml_libraries():
    """Pre-import TensorFlow and Basic Pitch on the main thread.

    TF's pybind11 C++ initialization is not thread-safe and crashes with
    "Unable to convert function return value to a Python type! () -> handle"
    if first imported inside a background thread.
    """
    try:
        import tensorflow  # noqa: F401
        from basic_pitch.inference import predict  # noqa: F401
        logger.info("Pre-warmed TensorFlow and Basic Pitch on main thread")
    except ImportError as e:
        logger.warning(f"Could not pre-warm ML libraries: {e}")


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)

    # Configuration
    default_uploads = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
    app.config['UPLOAD_FOLDER'] = os.environ.get('UPLOAD_FOLDER', default_uploads)
    app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload

    # Enable CORS for frontend (local dev + deployed Vercel)
    cors_origins = ['http://localhost:*', 'http://127.0.0.1:*']
    frontend_url = os.environ.get('FRONTEND_URL')
    if frontend_url:
        cors_origins.append(frontend_url)
    CORS(app, origins=cors_origins)

    # Pre-warm ML libraries on the main thread to avoid pybind11 threading issues
    _prewarm_ml_libraries()

    # Health check endpoint for Railway
    @app.route('/health')
    def health():
        return jsonify({'status': 'ok'}), 200

    # Register blueprints
    from app.routes import bp
    app.register_blueprint(bp)

    return app
