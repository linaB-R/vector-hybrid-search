import os
import pytest
from dotenv import load_dotenv

load_dotenv()

try:
	from openai import OpenAI
	_OPENAI_AVAILABLE = True
except Exception:
	_OPENAI_AVAILABLE = False


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_openai_key_present():
	assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY must be set"


@pytest.mark.skipif(not (_OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY")), reason="openai package or key not available")
def test_openai_models_list_smoke():
	client = OpenAI()
	# Lightweight smoke: list models; should not raise
	models = client.models.list()
	assert hasattr(models, "data")
