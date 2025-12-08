import os
from dotenv import load_dotenv

# Load .env file immediately upon import
load_dotenv()

def get_env_var(key: str, default: str = None) -> str:
    """Retrieves an environment variable."""
    return os.getenv(key, default)
