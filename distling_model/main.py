"""
Main Entry Point
━━━━━━━━━━━━━━━━
Provides CLI commands for the Distill AI Studio pipeline.

Usage:
  python main.py train       # Run full training pipeline
  python main.py agent       # Run agentic pipeline
  python main.py serve       # Start FastAPI backend
  python main.py status      # Show system status
"""

import sys
import os
import logging

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger("main")


def cmd_train():
    """Run the full training pipeline."""
    from training.pipeline import run_pipeline
    run_pipeline()


def cmd_agent():
    """Run the agentic pipeline orchestrator."""
    from agents import PipelineOrchestrator
    orchestrator = PipelineOrchestrator()
    orchestrator.run_full_pipeline()


def cmd_serve():
    """Start the FastAPI backend server."""
    import uvicorn
    import config
    logger.info("🚀 Starting Distill AI Studio Backend...")
    logger.info(f"   API:      http://localhost:{config.BACKEND_PORT}")
    logger.info(f"   Frontend: http://localhost:{config.BACKEND_PORT}/")
    logger.info(f"   Metrics:  http://localhost:{config.BACKEND_PORT}/metrics")
    logger.info(f"   Docs:     http://localhost:{config.BACKEND_PORT}/docs")

    # Change to project directory for relative imports
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=config.BACKEND_PORT,
        reload=False,
        log_level="info"
    )


def cmd_status():
    """Show system status."""
    import config
    import json

    print("\n" + "=" * 60)
    print("  DISTILL AI STUDIO — System Status")
    print("=" * 60)

    # Check data
    audio_classified = 0
    audio_unclassified = 0
    if os.path.isdir(config.CLASSIFIED_AUDIO_DIR):
        for cls in os.listdir(config.CLASSIFIED_AUDIO_DIR):
            cls_dir = os.path.join(config.CLASSIFIED_AUDIO_DIR, cls)
            if os.path.isdir(cls_dir):
                count = len([f for f in os.listdir(cls_dir)
                            if os.path.splitext(f)[1].lower() in {'.wav', '.mp3', '.flac'}])
                audio_classified += count
                print(f"  📂 {cls}: {count} files")

    if os.path.isdir(config.NON_CLASSIFIED_AUDIO_DIR):
        audio_unclassified = len([f for f in os.listdir(config.NON_CLASSIFIED_AUDIO_DIR)
                                   if os.path.splitext(f)[1].lower() in {'.wav', '.mp3', '.flac'}])

    print(f"\n  🎵 Classified Audio:    {audio_classified} files")
    print(f"  🎵 Non-classified Audio: {audio_unclassified} files")
    print(f"  📁 Audio Classes:        {config.AUDIO_CLASSES}")

    # Check models
    version_file = config.MODEL_VERSION_FILE
    if os.path.exists(version_file):
        with open(version_file, 'r') as f:
            registry = json.load(f)
        print(f"\n  🧠 Models Trained:       {len(registry.get('models', []))}")
        print(f"  🧠 Latest Version:       v{registry.get('latest_version', 0)}")
        for m in registry.get('models', []):
            print(f"     └─ v{m['version']} ({m['type']}) "
                  f"acc={m.get('accuracy', 0):.4f} — {m.get('timestamp', '')[:19]}")
    else:
        print("\n  🧠 No models trained yet")

    # Check feedback
    try:
        from feedback_loop import FeedbackDB
        db = FeedbackDB()
        print(f"\n  💬 Total Feedback:       {db.get_feedback_count()}")
        print(f"  💬 Unused (for retrain): {db.get_unused_count()}")
        print(f"  💬 Retrain Threshold:    {config.FEEDBACK_RETRAIN_THRESHOLD}")
        db.close()
    except Exception:
        print("\n  💬 Feedback DB: not initialized")

    # Check Ollama
    print(f"\n  🤖 Teacher Model:        {config.OLLAMA_MODEL_NAME}")
    try:
        import requests
        resp = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=3)
        if resp.status_code == 200:
            models = [m['name'] for m in resp.json().get('models', [])]
            available = any(config.OLLAMA_MODEL_NAME in m for m in models)
            print(f"  🤖 Ollama Status:        {'✅ Available' if available else '⚠️ Model not pulled'}")
        else:
            print(f"  🤖 Ollama Status:        ⚠️ Unexpected response")
    except Exception:
        print(f"  🤖 Ollama Status:        ❌ Not running")

    print("\n" + "=" * 60)
    print("  Commands:")
    print("    python main.py train   — Run training pipeline")
    print("    python main.py serve   — Start API server")
    print("    python main.py agent   — Run agentic pipeline")
    print("=" * 60 + "\n")


# ──────────────────────────────────────────────────────────────────────────────

COMMANDS = {
    "train": cmd_train,
    "agent": cmd_agent,
    "serve": cmd_serve,
    "status": cmd_status,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print("\nDistill AI Studio — CLI")
        print("Usage: python main.py <command>\n")
        print("Commands:")
        for name, fn in COMMANDS.items():
            print(f"  {name:12s}  {fn.__doc__.strip()}")
        print()
        sys.exit(1)

    COMMANDS[sys.argv[1]]()
