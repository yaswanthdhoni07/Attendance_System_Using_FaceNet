# app.py
from flask import Flask, jsonify
import argparse
from database import get_database
from routes.register_student import register_student_routes
from routes.take_attendance import take_attendance_routes
from routes.get_attendance import get_attendance_routes

app = Flask(__name__)

# ------------------ MongoDB Setup ------------------
db = get_database()

# ------------------ Register Routes ------------------
register_student_routes(app, db)
take_attendance_routes(app)
get_attendance_routes(app, db)

# ------------------ Utility Route ------------------
@app.route('/routes')
def list_routes():
    return jsonify([str(rule) for rule in app.url_map.iter_rules()])

# ------------------ Entry Point ------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Attendance System API")
    args = parser.parse_args()
    app.run(debug=True, port=5000)
