import os
import logging
from queue import Queue
import configparser

import utils

def initialize(): 
    global INSTALL_DIR,LOG_FILE,LOG_LEVEL,LOG_BACKUP_COUNT,OUTPUT,WORKERS,QUEUE_FILE,CONFIG_FILE,configfile,processed_count,matches_found,total_images,processed_files,stop_flag
    INSTALL_DIR=utils.get_main_dir()  
    LOG_LEVEL=logging.INFO
    CONFIG_FILE = os.path.join(INSTALL_DIR,"config.ini")
    configfile = configparser.ConfigParser()
 
    LOG_FILE = os.path.join(INSTALL_DIR,"logs", "FaceRecognition.log")
    QUEUE_FILE = os.path.join(INSTALL_DIR,"image_queue.txt")
    LOG_BACKUP_COUNT = 10  # Keep up to 10 backup logs

    stop_flag=False
    processed_count=0
    processed_files=[]
    total_images=0
    matches_found=[]
            
    load_config()
    WORKERS=int(configfile["Settings"]["workers"])
    OUTPUT=configfile["Settings"]["output_folder"]
    LOG_LEVEL=int(configfile["Settings"]["log_level"]) 
   
def load_config():
    """Load configuration from INI file."""
    if os.path.exists(CONFIG_FILE) and os.path.getsize(CONFIG_FILE) > 0:
        configfile.read(CONFIG_FILE)   
    else:
        # Create default config structure
        OUTPUT = os.path.join(INSTALL_DIR,"matched_faces")
        FACES_DIR = os.path.join(INSTALL_DIR,"faces")
        if os.cpu_count()>4:
            WORKERS=str(os.cpu_count()-2)
        else:
            WORKERS=str(os.cpu_count())    
        configfile["Settings"] = {
            "selected_image_path": FACES_DIR,
            "root_folder": "",
            "output_folder": OUTPUT,
            "workers": WORKERS,         #default to number of CPU
            "log_level": logging.INFO # https://docs.python.org/3/library/logging.html#logging-leve ls
        }
        save_config()
        load_config()# Load config at module import
            
def save_config():
    """Save the current configuration to the INI file."""
    with open(CONFIG_FILE, "w") as file:
        configfile.write(file)

