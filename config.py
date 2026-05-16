import os
from dotenv import load_dotenv

load_dotenv()

APP_TITLE = "PaperIQ – Research Paper Insight Analyzer"

MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb+srv://mannamhemanth02005_db_user:stM8isBj6AdicceU@cluster0.233ufvj.mongodb.net/paperiq?retryWrites=true&w=majority")
MONGODB_DB_NAME = os.environ.get("MONGODB_DB_NAME", "paperiq")
MONGODB_USERS_COLLECTION = os.environ.get("MONGODB_USERS_COLLECTION", "users")
MONGODB_ANALYSIS_COLLECTION = os.environ.get("MONGODB_ANALYSIS_COLLECTION", "analysis_history")
MONGODB_COUNTERS_COLLECTION = os.environ.get("MONGODB_COUNTERS_COLLECTION", "counters")

