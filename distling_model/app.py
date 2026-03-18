"""
Cat Emotion Analyzer - App Entry Point
========================================
Run this single file to start the full web application!

Usage:
    python app.py

Then open http://localhost:8000 in your browser.
Upload a cat image or audio to analyze its emotion.
"""

import os
import sys
import logging

# Ensure project root is on path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger("app")


def main():
    import config

    print()
    print("=" * 60)
    print("  CAT EMOTION ANALYZER - AI Studio")
    print("=" * 60)
    print()
    print(f"  Web UI:   http://localhost:{config.BACKEND_PORT}")
    print(f"  API Docs: http://localhost:{config.BACKEND_PORT}/docs")
    print(f"  Health:   http://localhost:{config.BACKEND_PORT}/health")
    print()
    print("  Upload a cat image or audio to analyze its emotion!")
    print("  Supported: JPG, PNG, WEBP, WAV, MP3, FLAC")
    print()
    print("=" * 60)
    print()

    # Change to project directory for relative imports
    os.chdir(ROOT_DIR)

    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=config.BACKEND_PORT,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()
