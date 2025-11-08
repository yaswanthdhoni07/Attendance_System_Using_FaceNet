# database.py
from pymongo import MongoClient

def get_database():
    client = MongoClient(
        "mongodb+srv://erlappa2025ds07_db_user:cdnsgEMOc8HkIGvX@cluster0.qexox7r.mongodb.net/smart_attendance?retryWrites=true&w=majority"
    )
    db = client["smart_attendance_db"]
    return db
