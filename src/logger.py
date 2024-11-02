import logging
import os
import sys
from datetime import datetime

DATE_FOLDER = f"{datetime.now().strftime('%Y_%m_%d')}"
logs_path=os.path.join(os.getcwd(),"logs", DATE_FOLDER) #,LOG_FILE
os.makedirs(logs_path,exist_ok=True)

LOG_FILE=f"{datetime.now().strftime('%Y_%m_%d__%H_%M_%S')}.log"
LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

if __name__ == "__main__":
    logging.info("Logging from logger file.")