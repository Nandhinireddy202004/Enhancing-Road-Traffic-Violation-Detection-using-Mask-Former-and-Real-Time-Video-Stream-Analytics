"""
app.py — TrafficSentinel Flask Application
Enhancing Road Traffic Violation Detection using Mask2Former and Real-Time Video Stream Analytics
"""

import os
import uuid
import shutil
import logging
import traceback
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, jsonify, flash

from traffic_core import TrafficViolationSystem, AppConfig

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "tvd-sentinel-2025")
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("TrafficSentinel")

AppConfig.ensure_directories()
tvs = TrafficViolationSystem()

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def safe_filename(original: str) -> str:
    ext = original.rsplit(".", 1)[-1].lower() if "." in original else "mp4"
    return f"{uuid.uuid4().hex}.{ext}"


def fmt_dt(iso: str) -> str:
    """Format ISO datetime for human display."""
    try:
        dt = datetime.fromisoformat(iso)
        return dt.strftime("%d %b %Y  %H:%M:%S")
    except Exception:
        return iso


def fmt_dt_parts(iso: str) -> dict:
    """Return date and time as separate strings for styled display."""
    try:
        dt = datetime.fromisoformat(iso)
        return {
            "date": dt.strftime("%d %b %Y"),
            "time": dt.strftime("%H:%M:%S"),
            "weekday": dt.strftime("%A"),
            "iso": iso,
        }
    except Exception:
        return {"date": iso, "time": "", "weekday": "", "iso": iso}


# ── Template context processor ────────────────────────────────────────────────

@app.context_processor
def inject_globals():
    sessions = tvs.history.load_index()
    return {
        "history_count": len(sessions),
        "now": datetime.now().strftime("%H:%M:%S"),
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Main upload / landing page."""
    return render_template("index.html", error=None)


@app.route("/upload", methods=["POST"])
def upload():
    """Accept video, run pipeline, redirect to result (PRG pattern)."""
    if "video" not in request.files:
        return render_template("index.html", error="No video file attached to request.")

    file = request.files["video"]
    if not file.filename:
        return render_template("index.html", error="No file selected.")
    if not allowed_file(file.filename):
        return render_template("index.html",
            error=f"Unsupported format. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}")

    # Save upload
    fname = safe_filename(file.filename)
    upload_path = os.path.join(AppConfig.UPLOAD_DIR, fname)
    try:
        file.save(upload_path)
    except Exception as exc:
        logger.error("Upload save failed: %s", exc)
        return render_template("index.html", error="Disk error — could not save upload.")

    # Run pipeline
    session_id     = uuid.uuid4().hex
    processed_name = f"proc_{fname}"
    processed_path = os.path.join(AppConfig.PROCESSED_DIR, processed_name)

    try:
        results = tvs.process_video(upload_path, processed_path, session_id=session_id)
        results["upload_video_url"]    = f"/static/uploads/{fname}"
        results["processed_video_url"] = f"/static/processed/{processed_name}"
        results["original_filename"]   = file.filename
        results["analysed_at_fmt"]     = fmt_dt(results.get("analysed_at", ""))
        results["analysed_at_parts"]   = fmt_dt_parts(results.get("analysed_at", ""))
        tvs.history.save_session(session_id, results)
        logger.info("Session %s complete — %d violations", session_id, results.get("total_violations", 0))
    except Exception as exc:
        logger.error("Pipeline error: %s\n%s", exc, traceback.format_exc())
        return render_template("index.html", error=f"Processing failed: {exc}")

    return redirect(url_for("result", session_id=session_id))


@app.route("/result/<session_id>")
def result(session_id: str):
    """Full analysis result page for one session."""
    data = tvs.history.load_session(session_id)
    if data is None:
        flash("Session not found.", "error")
        return redirect(url_for("index"))
    data["analysed_at_fmt"]   = fmt_dt(data.get("analysed_at", ""))
    data["analysed_at_parts"] = fmt_dt_parts(data.get("analysed_at", ""))
    return render_template("result.html", results=data)


@app.route("/history")
def history():
    """All past sessions, newest first."""
    sessions = tvs.history.load_index()
    for s in sessions:
        s["analysed_at_fmt"]   = fmt_dt(s.get("analysed_at", ""))
        s["analysed_at_parts"] = fmt_dt_parts(s.get("analysed_at", ""))
    return render_template("history.html", sessions=sessions)


@app.route("/history/delete/<session_id>", methods=["POST"])
def delete_session(session_id: str):
    tvs.history.delete_session(session_id)
    flash("Session removed from history.", "info")
    return redirect(url_for("history"))


@app.route("/reset", methods=["POST"])
def reset():
    """Clear temp files — history is preserved."""
    for d in [AppConfig.PROCESSED_DIR, AppConfig.UPLOAD_DIR, AppConfig.EVIDENCE_DIR]:
        if os.path.isdir(d):
            shutil.rmtree(d)
    AppConfig.ensure_directories()
    flash("Temporary files cleared. History preserved.", "info")
    return redirect(url_for("index"))


@app.route("/api/status")
def api_status():
    return jsonify({
        "status":        "operational",
        "timestamp":     datetime.utcnow().isoformat(),
        "models":        tvs.models_ready(),
        "sessions":      len(tvs.history.load_index()),
    })


# ── Entry ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info("TrafficSentinel on :%d", port)
    app.run(host="0.0.0.0", port=port, debug=False)