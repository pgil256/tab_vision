"""Flask application factory."""
import os
from flask import Flask
from flask_cors import CORS


def create_app():
    """Create and configure the Flask application."""
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
