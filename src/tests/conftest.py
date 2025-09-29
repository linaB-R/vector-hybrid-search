import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure project root is on sys.path so 'from src....' imports work under pytest
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load .env early for all tests
load_dotenv()


