# backend/routes/register_student.py
from flask import request, jsonify
import subprocess
import sys
import os

def register_student_routes(app, db):
    students_collection = db["students"]

    # ---------------- Correct project paths ----------------
    # Get SMART_ATTENDANCE project root
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # AI_model folder inside project root
    AI_DIR = os.path.join(PROJECT_ROOT, "AI_model")
    
    # Dataset folder inside AI_model
    DATASET_DIR = os.path.join(AI_DIR, "dataset")
    os.makedirs(DATASET_DIR, exist_ok=True)

    # Paths to AI scripts
    REGISTER_SCRIPT = os.path.join(AI_DIR, "register_student.py")
    EXTRACT_EMBEDDINGS_SCRIPT = os.path.join(AI_DIR, "extract_embeddings.py")
    TRAIN_CLASSIFIER_SCRIPT = os.path.join(AI_DIR, "train_classifier.py")

    @app.route('/register', methods=['POST'])
    def register_student_api():
        try:
            data = request.get_json()
            reg_no = data.get("reg_no")
            name = data.get("name")
            branch = data.get("branch")
            specialization = data.get("specialization")

            if not all([reg_no, name, branch, specialization]):
                return jsonify({"error": "Missing required details"}), 400

            # Check if student already exists
            if students_collection.find_one({"reg_no": reg_no}):
                return jsonify({"error": "Student already registered"}), 400

            # Save student details to MongoDB
            students_collection.insert_one({
                "reg_no": reg_no,
                "name": name,
                "branch": branch,
                "specialization": specialization
            })

            # Run registration script (captures images)
            subprocess.run([sys.executable, REGISTER_SCRIPT, "--reg_no", reg_no], check=True)
            
            # Extract embeddings
            subprocess.run([sys.executable, EXTRACT_EMBEDDINGS_SCRIPT], check=True)
            
            # Train classifier
            subprocess.run([sys.executable, TRAIN_CLASSIFIER_SCRIPT], check=True)

            # Count images saved
            person_dir = os.path.join(DATASET_DIR, reg_no)
            num_images = len(os.listdir(person_dir)) if os.path.exists(person_dir) else 0

            return jsonify({
                "status": "success",
                "message": f"Student {name} ({reg_no}) registered successfully!",
                "images_saved": num_images,
                "dataset_path": person_dir
            }), 200

        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
