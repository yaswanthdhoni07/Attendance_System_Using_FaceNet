# backend/routes/get_attendance.py

from flask import request, jsonify

def get_attendance_routes(app, db):
    """
    Registers the /get_attendance route on the given Flask app.
    Fetches attendance records based on subject and date range.
    """
    attendance_collection = db["attendance_records"]

    @app.route('/get_attendance', methods=['POST'])
    def get_attendance():
        try:
            data = request.get_json()
            subject = data.get("subject")
            from_date = data.get("from_date")
            to_date = data.get("to_date")

            # --- Build MongoDB query dynamically ---
            query = {}
            if subject:
                query["subject"] = {"$regex": f"^{subject}$", "$options": "i"}
            if from_date and to_date:
                query["date"] = {"$gte": from_date, "$lte": to_date}

            # --- Fetch matching records ---
            records = list(attendance_collection.find(query, {"_id": 0}))

            if not records:
                return jsonify({
                    "message": "No attendance records found.",
                    "subject": subject
                }), 404

            return jsonify({
                "count": len(records),
                "records": records
            }), 200

        except Exception as e:
            return jsonify({"error": str(e)}), 500
