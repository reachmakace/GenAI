import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')


# Backend parameter tuning
DEFAULT_RETRIEVAL_K = int(os.getenv('DEFAULT_RETRIEVAL_K', 5))
DEFAULT_CACHE_SIMILARITY_THRESHOLD = float(os.getenv('DEFAULT_CACHE_SIMILARITY_THRESHOLD', 0.92))

# Prompt templates
DEFAULT_SUMMARY_PROMPT = os.getenv('DEFAULT_SUMMARY_PROMPT', 'Summarize the following slide: {slide_text}')
DEFAULT_ANSWER_PROMPT = os.getenv('DEFAULT_ANSWER_PROMPT', 'You are a financial analyst. Given the following question: {question}, use the slides and images to provide a detailed answer.')

# Recency filter placeholder (can be extended)
DEFAULT_RECENCY_FILTER = os.getenv('DEFAULT_RECENCY_FILTER', None)
