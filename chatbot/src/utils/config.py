from os import getenv
# from zoneinfo import ZoneInfo
from dotenv import load_dotenv

load_dotenv()


# app config vars
DEBUG = True  # False
LOCAL = getenv("LOCAL")
LOCAL_DEBUG = DEBUG and LOCAL
MAX_GRAPH_EXECUTION_TIME = 25
MAX_TOKENS_AFTER_TRIMMING = 100                # used trim chat history
OPENAI_CLIENT_TIMEOUT = 5
RECURSION_LIMIT = 50                           # langgraph recursion limit
USE_LEGACY_AGENT = False
USE_LLAMA_INDEX = False
USE_PLAN_EXECUTE = False
WELCOME_PAGE_VALUE = """Welcome to Chattabot!\n
I'm your personal assistant, ready to answer your questions. I'm still \
a work-in-progress, but will get better over time the more I learn."""
TWILIO_ERROR_MESSAGE = """The message was not sent due to an error: {}"""


# error handling
RESPONSE_AT_RECURSION_ERROR = """Let me connect you to my colleague."""
RESPONSE_AT_MAX_EXECUTION_TIME = """I'm sorry. I'm having trouble \
understanding your question. Could you please rephrase that?"""
TIMEOUT_ERROR_MESSAGE = """I'm sorry. I didn't get that. \
Could you rephrase that?"""
API_ERROR_MESSAGE = """Sorry. We are having a technical difficulty. \
Please try again later."""
OTHER_ERROR_MESSAGE = """Sorry. We are having a technical difficulty. \
I will connect you to my colleague."""


# DB config vars
CHECKPOINT_INDEX_NAME = "for_deletion"
INDEX_NAME = "business_description"
RUN_EXACT_NEAREST_NEIGHBOR_VECTOR_SEARCH = True
RETRIEVER_POST_FILTER_MIN_SIMILARITY_SCORE = 0.60
TTL_INDEX_KEY = "created_at"
TTL_EXPIRE_AFTER_SECONDS = 180

# LLM config vars
EMBEDDING_MODEL_NAME = "text-embedding-3-large"   # "text-embedding-ada-002"
MAX_RETRIES = 1                             # the default is 2 in ChatOpenAI
AGENT_MODEL_NAME = "gpt-4-0125-preview"     # "gpt-4o-2024-08-06" or "gpt-4o"
AGENT_TEMPERATURE = 0.1
EVALUATOR_MODEL_NAME = "gpt-4o-2024-08-06"  # "o1-preview-2024-09-12"
EVALUATOR_TEMPERATURE = 0.1
O1_MODEL_NAME = "o1-mini-2024-09-12"        # "o1-preview-2024-09-12"
O1_MODEL_TEMPERATURE = 1
PLANNER_MODEL_NAME = "gpt-4o-2024-08-06"
PLANNER_TEMPERATURE = 0.0
NODE_ACTION_MODEL_NAME = "gpt-4o-2024-08-06"   # "gpt-4o-mini"
NODE_ACTION_TEMPERATURE = 0.1
CHAT_HISTORY_TRIMMER_MODEL_NAME = AGENT_MODEL_NAME
# TIMEZONE = ZoneInfo("US/Eastern")                # for datetime tool
TIMEZONE = None
