# get_attendance.py
from flask import request, jsonify

def register_get_attendance_route(app, db):
    attendance_collection = db["attendance_records"]

    @app.route('/get_attendance', methods=['POST'])
    def get_attendance():
        """
        Fetch attendance records filtered by subject and/or date range.
        Example input:
        {
            "subject": "Advance Power systems",
            "from_date": "2025-11-01",
            "to_date": "2025-11-10"
        }
        """
        try:
            data = request.get_json()
            subject = data.get("subject")
            from_date = data.get("from_date")
            to_date = data.get("to_date")

            query = {}

            # ✅ Subject filter (case-insensitive)
            if subject:
                query["subject"] = {"$regex": f"^{subject}$", "$options": "i"}

            # ✅ Date range filter (string-safe)
            if from_date and to_date:
                query["date"] = {"$gte": from_date, "$lte": to_date}
            elif from_date:  # only from_date
                query["date"] = {"$gte": from_date}
            elif to_date:  # only to_date
                query["date"] = {"$lte": to_date}

            # --- Query DB ---
            records = list(attendance_collection.find(query, {"_id": 0}))

            if not records:
                return jsonify({
                    "message": "No attendance records found in the given range.",
                    "subject": subject
                }), 404

            return jsonify({
                "count": len(records),
                "records": records
            }), 200

        except Exception as e:
            return jsonify({"error": str(e)}), 500
