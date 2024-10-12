CHUNK_OVERLAP = 10      # 15 is also good
CHUNK_SIZE = 256        # 100 seems optimal for MongoDB, 512 is also feasible
CHUNK_SEPARATOR = "\n\n"
DEBUG = False
INDEX_NAME = "business_description"
LOCAL = False
MODEL_NAME = "text-embedding-3-large"   # "text-embedding-ada-002"
USE_RECURSIVE_SPLITTER = False