"""Flask application factory."""
from flask import Flask
from flask_cors import CORS


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    CORS(app)

    # Register blueprints
    from app.routes import bp
    app.register_blueprint(bp)

    return app
