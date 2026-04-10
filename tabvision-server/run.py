"""Entry point for the TabVision Flask server."""
from app import create_app

app = create_app()

if __name__ == '__main__':
    # use_reloader=False prevents Flask from forking a child process,
    # which causes pybind11 errors in TensorFlow/MediaPipe:
    # "Unable to convert function return value to a Python type! The signature was () -> handle"
    app.run(debug=True, port=5000, use_reloader=False)
