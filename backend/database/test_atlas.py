
from datetime import datetime
from flask import Flask, request, jsonify
import subprocess
import argparse
from pymongo import MongoClient

app = Flask(__name__)

# --- MongoDB Atlas Connection ---
MONGO_URI ="mongodb+srv://erlappa2025ds07_db_user:cdnsgEMOc8HkIGvX@cluster0.qexox7r.mongodb.net/smart_attendance?retryWrites=true&w=majority"
client = MongoClient(MONGO_URI)
db = client["smart_attendance_db"]
students_collection = db["students"]

# --- Paths to your scripts ---
REGISTER_SCRIPT = r"../AI_model/register_student.py"
TRAIN_SCRIPT = r"../AI_model/train_model.py"

@app.route('/register', methods=['POST'])
def register_student_api():
    try:
        data = request.get_json()

        reg_no = data.get("reg_no")
        name = data.get("name")
        branch = data.get("branch")
        specialization = data.get("specialization")

        if not all([reg_no, name, branch, specialization]):
            return jsonify({"error": "Missing required student details"}), 400

        # Check for duplicate reg_no
        if students_collection.find_one({"reg_no": reg_no}):
            return jsonify({"error": "Student already registered"}), 400

        # Insert student into MongoDB
        students_collection.insert_one({
            "reg_no": reg_no,
            "name": name,
            "branch": branch,
            "specialization": specialization
        })

        # Run model registration and training
        run_registration_and_training(reg_no)

        return jsonify({
            "status": "success",
            "message": f"Student {name} ({reg_no}) registered successfully!"
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


def run_registration_and_training(reg_no):
    """Run student registration (face capture) + model training."""
    print(f"🔹 Registering and training for {reg_no}")
    subprocess.run(["python", REGISTER_SCRIPT, "--reg_no", reg_no], check=True)
    subprocess.run(["python", TRAIN_SCRIPT], check=True)
    print(f"✅ Training complete for {reg_no}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register student and train model")
    parser.add_argument("--reg_no", type=str, help="Student registration number")
    args = parser.parse_args()

    if args.reg_no:
        run_registration_and_training(args.reg_no)
    else:
        app.run(debug=True, port=5000)
