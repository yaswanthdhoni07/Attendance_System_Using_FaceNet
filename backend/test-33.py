from flask import Flask, request, jsonify
import subprocess
import os
import sys
import argparse
from pymongo import MongoClient
import datetime
import json
from get_attendance import register_get_attendance_route


app = Flask(__name__)

# ------------------ MongoDB Setup ------------------
client = MongoClient(
    "mongodb+srv://erlappa2025ds07_db_user:cdnsgEMOc8HkIGvX@cluster0.qexox7r.mongodb.net/smart_attendance?retryWrites=true&w=majority"
)
db = client["smart_attendance_db"]
students_collection = db["students"]
attendance_collection = db["attendance_records"]

# ------------------ Paths to Scripts ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REGISTER_SCRIPT = os.path.join(BASE_DIR, "../AI_model/register_student.py")
EXTRACT_EMBEDDINGS_SCRIPT = os.path.join(BASE_DIR, "../AI_model/extract_embeddings.py")
TRAIN_CLASSIFIER_SCRIPT = os.path.join(BASE_DIR, "../AI_model/train_classifier.py")
TAKE_ATTENDANCE_SCRIPT = os.path.join(BASE_DIR, "../AI_model/test_att.py")
RECOGNIZED_FILE = os.path.join(BASE_DIR, "../AI_model/recognized_students.json")  # output file

# ======================================================
# 📌 Route: Register Student and Train Model
# ======================================================
@app.route('/register', methods=['POST'])
def register_student_api():
    """API endpoint for frontend form submission."""
    try:
        data = request.get_json()

        reg_no = data.get("reg_no")
        name = data.get("name")
        branch = data.get("branch")
        specialization = data.get("specialization")

        # --- Basic validation ---
        if not all([reg_no, name, branch, specialization]):
            return jsonify({"error": "Missing required student details"}), 400

        # --- Step 1: Save to MongoDB ---
        existing_student = students_collection.find_one({"reg_no": reg_no})
        if existing_student:
            return jsonify({"error": "Student already registered"}), 400

        students_collection.insert_one({
            "reg_no": reg_no,
            "name": name,
            "branch": branch,
            "specialization": specialization
        })

        # --- Step 2: Run registration + training chain ---
        run_registration_and_training_chain(reg_no)

        return jsonify({
            "status": "success",
            "message": f"Student {name} ({reg_no}) registered and model retrained successfully!"
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


def run_registration_and_training_chain(reg_no):
    """Runs registration, then embeddings extraction, then classifier training sequentially."""
    print(f"🔹 Starting registration for student: {reg_no}")

    # Step 1: Run face registration (capture images)
    subprocess.run([sys.executable, REGISTER_SCRIPT, "--reg_no", reg_no], check=True)

    # Step 2: Extract embeddings (wait until completion)
    print("🔹 Extracting embeddings...")
    subprocess.run([sys.executable, EXTRACT_EMBEDDINGS_SCRIPT], check=True)

    # Step 3: Train classifier after embeddings are ready
    print("🔹 Training classifier...")
    subprocess.run([sys.executable, TRAIN_CLASSIFIER_SCRIPT], check=True)

    print(f"✅ Registration, embedding extraction, and classifier training completed for {reg_no}")

register_get_attendance_route(app, db)
# ======================================================
# 📌 Route: Take Attendance (with Subject)
# ======================================================
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
            cwd=os.path.dirname(TAKE_ATTENDANCE_SCRIPT)
        )

        return jsonify({"status": "success", "message": f"Attendance taken for {subject}"}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ======================================================
# 📌 Utility: List all routes
# ======================================================
@app.route('/routes')
def list_routes():
    routes = [str(rule) for rule in app.url_map.iter_rules()]
    return jsonify(routes)


# ======================================================
# 📌 Entry Point
# ======================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register or take attendance")
    parser.add_argument("--reg_no", type=str, help="Student registration number")
    args = parser.parse_args()

    if args.reg_no:
        run_registration_and_training_chain(args.reg_no)
    else:
        app.run(debug=True, port=5000)
