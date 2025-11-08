# routes/take_attendance.py
from flask import request, jsonify
import subprocess
import sys
import os

def take_attendance_routes(app):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    AI_DIR = os.path.join(BASE_DIR, "../../AI_model")
    TAKE_ATTENDANCE_SCRIPT = os.path.join(AI_DIR, "face_recognition.py")

    @app.route('/take_attendance', methods=['POST'])
    def take_attendance():
        try:
            data = request.get_json()
            subject = data.get("subject")

            if not subject:
                return jsonify({"error": "Missing subject"}), 400

            subprocess.run(
                [sys.executable, TAKE_ATTENDANCE_SCRIPT, "--subject", subject],
                check=True,
                cwd=AI_DIR
            )

            return jsonify({
                "status": "success",
                "message": f"Attendance taken for {subject}"
            }), 200

        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
