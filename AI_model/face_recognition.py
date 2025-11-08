import cv2
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
import joblib
import datetime
from collections import deque, Counter
import csv
from pymongo import MongoClient
import json
import argparse

# ------------------- MongoDB setup -------------------
client = MongoClient(
    "mongodb+srv://erlappa2025ds07_db_user:cdnsgEMOc8HkIGvX@cluster0.qexox7r.mongodb.net/smart_attendance?retryWrites=true&w=majority"
)
db = client["smart_attendance_db"]
students_collection = db["students"]
attendance_collection = db["attendance_records"]

# ------------------- Device setup -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------- Models -------------------
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, device=device)
facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
classifier = joblib.load("./model/svm_classifier.joblib")
label_encoder = joblib.load("./model/label_encoder.joblib")

# ------------------- Globals -------------------
attendance = {}
prediction_buffer = deque(maxlen=7)
CONFIDENCE_THRESHOLD = 0.8
RECOGNIZED_JSON = "./recognized_students.json"


# ------------------- Mark Attendance -------------------
def mark_attendance(student_id, subject):
    """Mark attendance for recognized student."""
    now = datetime.datetime.now()
    if student_id not in attendance:
        day = now.strftime("%A")
        date_part = now.strftime("%Y-%m-%d")
        time_part = now.strftime("%H:%M:%S")

        # Fetch student details from DB
        student = students_collection.find_one(
            {"$or": [
                {"reg_no": student_id},
                {"Reg_no": student_id},
                {"REG_NO": student_id}
            ]}
        )

        if not student:
            print(f"⚠️ Student not found in DB: {student_id}")
            name = None
            branch = None
            specialization = None
        else:
            name = student.get("name") or student.get("Name")
            branch = student.get("branch") or student.get("Branch")
            specialization = (
                student.get("specialization")
                or student.get("Dept")
                or student.get("department")
            )

        record = {
            "reg_no": student_id,
            "name": name,
            "branch": branch,
            "specialization": specialization,
            "subject": subject,
            "day": day,
            "date": date_part,
            "time": time_part,
        }

        try:
            attendance_collection.insert_one(record)
            attendance[student_id] = record
            print(f"✅ Stored in DB: {student_id} ({name}) on {day} {date_part} {time_part}")
        except Exception as e:
            print(f"❌ Failed to store {student_id} in DB: {e}")


# ------------------- Main Recognition -------------------
def main(subject):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera not detected!")
        return

    recognized_students = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        face_tensor = mtcnn(img)

        if face_tensor is not None:
            face_tensor = face_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                embedding = facenet(face_tensor).cpu().numpy()

            # Predict class
            probs = classifier.predict_proba(embedding)
            max_prob = np.max(probs)
            pred_idx = classifier.predict(embedding)[0]
            pred_id = label_encoder.inverse_transform([pred_idx])[0]

            if max_prob > CONFIDENCE_THRESHOLD:
                prediction_buffer.append(pred_id)
            else:
                prediction_buffer.append("Unknown")

            # Majority vote smoothing
            most_common_pred = Counter(prediction_buffer).most_common(1)[0][0]

            if most_common_pred != "Unknown":
                mark_attendance(most_common_pred, subject)
                recognized_students.add(most_common_pred)
                label_text = f"ID: {most_common_pred} ({max_prob*100:.1f}%)"
                color = (0, 255, 0)
            else:
                label_text = "Unknown"
                color = (0, 0, 255)
        else:
            prediction_buffer.append("Unknown")
            label_text = "No Face"
            color = (0, 0, 255)

        # Display frame
        cv2.putText(frame, label_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("FaceNet Attendance (Smoothed)", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save attendance to CSV
    with open("attendance.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Reg_No", "Name", "Branch", "Specialization", "Subject", "Day", "Date", "Time"])
        for s_id, record in attendance.items():
            writer.writerow([
                record.get("reg_no"),
                record.get("name"),
                record.get("branch"),
                record.get("specialization"),
                record.get("subject"),
                record.get("day"),
                record.get("date"),
                record.get("time"),
            ])
    print("📁 Attendance saved to attendance.csv")

    # Save recognized student IDs to JSON
    recognized_list = list(recognized_students)
    with open(RECOGNIZED_JSON, "w") as jf:
        json.dump({"recognized_students": recognized_list}, jf, indent=4)
    print(f"🧾 Recognized students saved to {RECOGNIZED_JSON}")


# ------------------- Entry Point -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Recognition Attendance")
    parser.add_argument("--subject", type=str, required=True, help="Subject name for attendance")
    args = parser.parse_args()
    main(args.subject)
