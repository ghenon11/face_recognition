import sys,os,shutil,logging
from logging.handlers import RotatingFileHandler
import threading
from threading import Timer
from contextlib import contextmanager

import config

def ensure_directories(one_directory):
    try:
        if os.path.isdir(one_directory):
            result=os.makedirs(one_directory, exist_ok=True)
        else:            
            result=os.makedirs(os.path.basename(os.path.dirname(one_directory)), exist_ok=True)
    except Exception as e:
        print(f"Error ensure directories: {e}")
    return result
        
def init_logging():   
    # Configure logging
    #logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s(%(thread)d) - %(levelname)s - %(message)s")
    # Rotate the log file at each startup
    ensure_directories(config.LOG_FILE)
    if os.path.exists(config.LOG_FILE):
        try:
            for i in range(config.LOG_BACKUP_COUNT, 0, -1):
                old_log = f"{config.LOG_FILE}.{i}"
                older_log = f"{config.LOG_FILE}.{i - 1}" if i > 1 else config.LOG_FILE
                if os.path.exists(old_log):
                    os.remove(old_log)  # Remove the target log if it already exists
                if os.path.exists(older_log):
                    os.rename(older_log, old_log)
        except Exception as e:
            print(f"Error rotating logs at startup: {e}")
            
    logging.basicConfig(
        format='%(asctime)s::%(processName)s(%(process)d)::%(levelname)s::%(message)s',
        level=config.LOG_LEVEL,
        handlers=[
            RotatingFileHandler(config.LOG_FILE, maxBytes=50 * 1024 * 1024, backupCount=config.LOG_BACKUP_COUNT)
        ]
    )
    
def path_exists(a_path):
    return os.path.exists(a_path)
 
def get_main_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    if os.path.dirname(sys.argv[0]):
        return os.path.dirname(sys.argv[0]) # name of file
    return os.path.dirname(__file__)
    
def has_enough_space(folder):
    """Check if there is at least 1GB of free space."""
    if os.path.isfile(folder) or os.path.isdir(folder):
        total, used, free = shutil.disk_usage(folder)
        return free >= config.MIN_FREE_SPACE_BYTES
    return True
    


def remove_special_characters(character):
    if character in [' ','!', '"', '#', '$', '%', '&', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[',']', '^', '_', '`', '{', '|', '}', '~']:
        return True
    if character.isalnum() :
        return True
    else:
        return False

import threading
from contextlib import contextmanager        
class RepeatedTimer:
    def __init__(self, interval, function, *args, **kwargs):
        self._timer = None
        self.interval = interval
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.is_running = False
        self.start()

    def _run(self):
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)

    def start(self):
        if not self.is_running:
            self._timer = Timer(self.interval, self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        if self._timer:
            self._timer.cancel()
            self.is_running = False