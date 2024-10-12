##############################################################################
##############################################################################
# ####### SCRIPT TO BE PERIODICALLY RUN TO REMOVE STALE CHAT HISTORY ####### #
##############################################################################
##############################################################################

from datetime import datetime, UTC
from os import getenv

from apscheduler.schedulers.background import BackgroundScheduler

from pymongo.mongo_client import MongoClient

DATABASE_NAME = getenv("BUSINESS_NAME")
MONGO_URI = getenv("CHAT_HISTORY_MONGO_URI")


def remove_empty_collections() -> None:
    client = MongoClient(MONGO_URI)
    db = client[DATABASE_NAME]
    for col_name in list(db.list_collection_names()):
        if next(db[col_name].find(), None) is None:
            db[col_name].drop()


if __name__ == '__main__':
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        func=remove_empty_collections,
        trigger="interval",
        startdate=datetime(2024, 9, 10, 9, 9, 9, tzinfo=UTC),
        weeks=1
    )
    # scheduler.start()
    # try:
    #     # This is here to simulate application activity
    #     # (which keeps the main thread alive).
    #     while True:
    #         import time
    #         time.sleep(2)
    # except (KeyboardInterrupt, SystemExit):
    #     # Not strictly necessary if daemonic mode is enabled
    #     # but should be done if possible
    #     scheduler.shutdown()
