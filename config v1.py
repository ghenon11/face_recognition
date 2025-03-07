import os
import logging
from queue import Queue

import utils

def initialize(): 
    global INSTALL_DIR,LOG_FILE,LOG_LEVEL,LOG_BACKUP_COUNT,OUTPUT,WORKERS,QUEUE_FILE,FACES_DIR,processed_count,matches_found,total_images,processed_files,stop_flag
    INSTALL_DIR=utils.get_main_dir()    
    LOG_LEVEL=logging.INFO
    LOG_FILE = os.path.join(INSTALL_DIR,"logs", "FaceRecognition.log")
    FACES_DIR=os.path.join(INSTALL_DIR,"faces")
    QUEUE_FILE = os.path.join(INSTALL_DIR,"image_queue.txt")
    LOG_BACKUP_COUNT = 10  # Keep up to 10 backup logs
    #OUTPUT = os.path.join(INSTALL_DIR,"matched_faces")
    OUTPUT = "d:\matched_faces"
    WORKERS = 4
    stop_flag=False
    processed_count=0
    processed_files=[]
    total_images=0
    matches_found=[]


