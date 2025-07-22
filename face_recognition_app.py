import os,time
import shutil
import logging, traceback
import threading, multiprocessing, setproctitle
import random
import concurrent.futures
import face_recognition
import utils, config
import numpy as np
from queue import Queue
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog
import customtkinter as ctk
from PIL import Image
import sqlite3
import hashlib

lock = threading.Lock()
DB_PATH = "face_recognition.db"

def init_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        # Table for file paths (can have same hash in different locations)
        c.execute("""
            CREATE TABLE IF NOT EXISTS file_paths (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT UNIQUE
            )
        """)
        # Table for images (hash and metadata, not tied to path)
        c.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hash TEXT,
                num_faces INTEGER
            )
        """)
        # Table for mapping file path to image hash
        c.execute("""
            CREATE TABLE IF NOT EXISTS file_image_map (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path_id INTEGER,
                image_id INTEGER,
                FOREIGN KEY(file_path_id) REFERENCES file_paths(id),
                FOREIGN KEY(image_id) REFERENCES images(id),
                UNIQUE(file_path_id, image_id)
            )
        """)
        # Table for matches (now with person_id instead of encoding)
        c.execute("""
            CREATE TABLE IF NOT EXISTS matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER,
                person_id INTEGER,
                FOREIGN KEY(image_id) REFERENCES images(id),
                FOREIGN KEY(person_id) REFERENCES persons(id)
            )
        """)
        # Table for persons
        c.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE
            )
        """)
        # Table for known images (used for known encodings)
        c.execute("""
            CREATE TABLE IF NOT EXISTS known_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER,
                image_id INTEGER,
                FOREIGN KEY(person_id) REFERENCES persons(id),
                FOREIGN KEY(image_id) REFERENCES images(id)
            )
        """)
        # Table for all encodings per image
        c.execute("""
            CREATE TABLE IF NOT EXISTS image_encodings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER,
                encoding BLOB,
                UNIQUE(image_id, encoding),
                FOREIGN KEY(image_id) REFERENCES images(id)
            )
        """)
        conn.commit()
    except Exception as e:
        logging.error(f"SQLite error in init_db: {e}")
    finally:
        conn.close()

def hash_image(file_path):
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def get_or_create_file_path(path):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT id FROM file_paths WHERE path=?", (path,))
        row = c.fetchone()
        if row:
            file_path_id = row[0]
        else:
            c.execute("INSERT INTO file_paths (path) VALUES (?)", (path,))
            conn.commit()
            file_path_id = c.lastrowid
        conn.close()
        return file_path_id
    except Exception as e:
        logging.error(f"SQLite error in get_or_create_file_path: {e}")
        return None

def get_or_create_image(hash_val, num_faces):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT id FROM images WHERE hash=?", (hash_val,))
        row = c.fetchone()
        if row:
            image_id = row[0]
        else:
            c.execute("INSERT INTO images (hash, num_faces) VALUES (?, ?)", (hash_val, num_faces))
            conn.commit()
            image_id = c.lastrowid
        conn.close()
        return image_id
    except Exception as e:
        logging.error(f"SQLite error in get_or_create_image: {e}")
        return None

def map_file_to_image(file_path_id, image_id):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT id FROM file_image_map WHERE file_path_id=? AND image_id=?", (file_path_id, image_id))
        row = c.fetchone()
        if not row:
            c.execute("INSERT INTO file_image_map (file_path_id, image_id) VALUES (?, ?)", (file_path_id, image_id))
            conn.commit()
        conn.close()
    except Exception as e:
        logging.error(f"SQLite error in map_file_to_image: {e}")

def insert_image_encoding(image_id, encoding):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT id FROM image_encodings WHERE image_id=? AND encoding=?", (image_id, encoding.tobytes()))
        exists = c.fetchone()
        if not exists:
            c.execute("INSERT INTO image_encodings (image_id, encoding) VALUES (?, ?)", (image_id, encoding.tobytes()))
            conn.commit()
        conn.close()
    except Exception as e:
        logging.error(f"SQLite error in insert_image_encoding: {e}")

def insert_match(image_id, person_id):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO matches (image_id, person_id) VALUES (?, ?)", (image_id, person_id))
        conn.commit()
        conn.close()
    except Exception as e:
        logging.error(f"SQLite error in insert_match: {e}")

class FaceRecognitionApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Face Recognition App")
        width = 800
        height = 680
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width / 2) - (width / 2)
        y = (screen_height / 2) - (height / 2)
        self.geometry(f'{width}x{height}+{int(x)}+{int(y)}')
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.protocol("WM_DELETE_WINDOW", self.confirm_close)
        
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)
        for i in range(9):
            self.main_frame.grid_rowconfigure(i, weight=1)
        
        self.main_frame.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")
        self.label = ctk.CTkLabel(self.main_frame, text="Face recognition")
        self.label.grid(row=0, column=0, padx=10, pady=5, columnspan=2)

        self.folder_button = ctk.CTkButton(self.main_frame, text="Select Root Folder", command=self.select_root_folder)
        self.folder_button.grid(row=2, column=0, padx=10, pady=5)
        self.label_root = ctk.CTkLabel(self.main_frame)
        self.label_root.grid(row=2, column=1, padx=10, pady=5)
        self.label_root_var = ctk.StringVar(value="Select Root Folder")
        self.label_root.configure(textvariable=self.label_root_var)
       
        self.output_button = ctk.CTkButton(self.main_frame, text="Select Output Folder", command=self.select_output_folder)
        self.output_button.grid(row=3, column=0, padx=10, pady=5)
        self.label_output = ctk.CTkLabel(self.main_frame)
        self.label_output.grid(row=3, column=1, padx=10, pady=5)
        self.label_output_var = ctk.StringVar(value="Set output path")
        self.label_output.configure(textvariable=self.label_output_var)
        
        self.compare_button = ctk.CTkButton(self.main_frame, text="Compare Faces", command=self.start_comparison, state="disabled")
        self.compare_button.grid(row=4, column=0, padx=10, pady=5,columnspan=2)

        self.progress_bar = ctk.CTkProgressBar(self.main_frame,orientation="horizontal",width=500)
        self.progress_bar.set(0)
        self.progress_bar.grid(row=5, column=0, padx=10, pady=5,columnspan=2)

        self.exit_button = ctk.CTkButton(self.main_frame, text="Exit Application", fg_color="red", hover_color="darkred", command=self.exit_app).grid(row=6, column=0, padx=10, pady=5,columnspan=2)

        self.image_label = ctk.CTkLabel(self.main_frame, text="",width=200, height=200)
        self.image_label.grid(row=7, column=0, padx=10, pady=5,columnspan=2)

        self.match_image_label = ctk.CTkLabel(self.main_frame, text="",width=200, height=200)
        self.match_image_label.grid(row=8, column=0, padx=10, pady=5,columnspan=2)
        
        self.queue_root_button = ctk.CTkButton(
            self.main_frame,
            text="Queue Only Root Folder Images",
            command=self.queue_only_root_folder
        )
        self.queue_root_button.grid(row=9, column=0, padx=10, pady=5, columnspan=2)
        
        self.root_folder = None
        self.matching_folders = []
        self.image_queue = Queue()
        self.current_image = None
        
        self.root_folder = config.configfile.get("Settings", "root_folder", fallback="")
        self.output_folder = config.configfile.get("Settings", "output_folder", fallback="")
        self.selected_image_path = config.configfile.get("Settings", "selected_image_path", fallback="")

        
        if self.root_folder:
            self.label_root_var.set(self.root_folder)
        if self.output_folder:
            self.label_output_var.set(self.output_folder)
        if self.root_folder and self.output_folder:
            self.label.configure(text=f"Configuration loaded")
            self.compare_button.configure(state="normal")

        self.update_random_face()
        self.update_ui()

    def update_random_face(self):
        faces_dir = self.selected_image_path
        if not os.path.exists(faces_dir):
            logging.warning("Faces directory not found!")
            return
        image_files = [os.path.join(faces_dir, f) for f in os.listdir(faces_dir) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
        if not image_files:
            logging.warning("No images found in faces directory!")
            return
        random_image = Path(random.choice(image_files))
        self.display_image(random_image, self.image_label)
        self.after(10000, self.update_random_face) 
        
    def select_image(self):
        folder = filedialog.askdirectory()
        if folder:
            self.selected_image_path = folder
            self.label.configure(text=f"Selected: {os.path.basename(folder)}")
            self.img_path.set(self.selected_image_path)
            config.configfile.set("Settings", "selected_image_path", folder)
            config.save_config()
            if self.root_folder:
                self.compare_button.configure(state="normal")

    def display_image(self, file_path, label):
        logging.info(f"Displaying {file_path}")
        f = open(file_path, 'rb')
        try:
            image = Image.open(f)
            aspect_ratio = image.width / image.height
            new_width = 200
            new_height = int(new_width / aspect_ratio)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            photo = ctk.CTkImage(light_image=image, dark_image=image, size=(new_width, new_height))
            label.configure(image=photo)
            label.image = photo
        except Exception as e:
            logging.warning(f"Error displaying '{file_path}': {e}")
            
    def select_root_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.root_folder = folder
            self.label_root_var.set(self.root_folder)
            config.configfile.set("Settings", "root_folder", folder)
            config.save_config()
            self.matching_folders = []
            for dirpath, dirnames, filenames in os.walk(folder):
                self.matching_folders.append(dirpath)
            self.label.configure(text=f"Matching Folders: {len(self.matching_folders)} found")
            logging.info(f"Matching Folders: {self.matching_folders}")
            if self.selected_image_path:
                self.compare_button.configure(state="normal") 
    
    def select_output_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.output_folder = folder
            self.label_output_var.set(self.output_folder)
            config.configfile.set("Settings", "output_folder", folder)
            config.save_config()
                
    def confirm_close(self):
        if messagebox.askyesno("Confirm Close", "Are you sure you want to close?"):
            self.exit_app()

    def exit_app(self):
        logging.info("Exiting Application")
        config.stop_flag=True
        time.sleep(10)
        self.quit()

    def count_images_in_folders(self):
        total_images = 0
        for folder in self.matching_folders:
            try:
                image_files = [f for f in os.listdir(folder) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
                total_images += len(image_files)
            except Exception as e:
                logging.warning(f"Error accessing folder '{folder}': {e}")
                continue
        return total_images

    def queue_images(self, folders=None):
        logging.info("Building image list")
        queued_images = []
        if folders is None:
            folders = self.matching_folders
        for folder in folders:
            try:
                image_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
                for img in image_files:
                    formatted_path = Path(img)
                    hash_val = hash_image(img)
                    file_path_id = get_or_create_file_path(str(formatted_path))
                    try:
                        conn = sqlite3.connect(DB_PATH)
                        c = conn.cursor()
                        c.execute("SELECT images.id FROM images "
                                  "JOIN file_image_map ON images.id = file_image_map.image_id "
                                  "WHERE file_image_map.file_path_id=? AND images.hash=?", (file_path_id, hash_val))
                        row = c.fetchone()
                        if row:
                            image_id = row[0]
                            # Hash exists for this file path, skip face_recognition
                            num_faces = None
                            encodings = []
                            logging.info(f"Image queued: {img} (encoding already exists for this path, hash unchanged)")
                        else:
                            # Hash is new or changed for this file path, process image
                            image = face_recognition.load_image_file(img)
                            encodings = face_recognition.face_encodings(image)
                            num_faces = len(encodings)
                            image_id = get_or_create_image(hash_val, num_faces)
                            logging.info(f"Image queued: {img} (encoding done, hash new or changed for this path)")
                        conn.close()
                    except Exception as e:
                        logging.warning(f"Error loading image {img}: {e}")
                        num_faces = 0
                        encodings = []
                        image_id = get_or_create_image(hash_val, num_faces)
                    map_file_to_image(file_path_id, image_id)
                    if encodings:
                        for encoding in encodings:
                            insert_image_encoding(image_id, encoding)
                    self.image_queue.put(formatted_path)
                    queued_images.append(str(formatted_path))
            except Exception as e:
                logging.warning(f"Error accessing folder '{folder}': {e}")
                continue
        logging.info(f"Queued {len(queued_images)} images for processing.")

    def start_comparison(self):
        thread = threading.Thread(target=self.compare_faces)
        thread.start()
              
    def compare_faces(self):
        if not self.selected_image_path:
            messagebox.showerror("Error", "Please select an image and a root folder with matching subfolders!")
            return
        if not self.output_folder or not os.path.exists(self.output_folder):
            messagebox.showerror("Error", "Please select a folder to hold images that matches !")
            return
        output_folder = self.output_folder
        os.makedirs(output_folder, exist_ok=True)
        self.label.configure(text=f"Starting image comparison")
        logging.info("Building list of files")
        self.queue_images()
        config.total_images = self.image_queue.qsize()
        logging.info(f"{config.total_images} images to process")
        config.processed_count = 0
        logging.info(f"Encoding known faces")
        known_encodings, known_person_ids = load_known_encodings_from_db()
        if not known_encodings:
            raise ValueError("No face found in the selected image!")
        known_encodings_list = [enc.tolist() for enc in known_encodings]
        logging.info(f"Start processing with {config.WORKERS} workers")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=config.WORKERS, initializer=worker_init, initargs=(os.getpid(),)
        ) as executor:
            futures = {}
            while not self.image_queue.empty():
                file_path = self.image_queue.get()
                future = executor.submit(process_image, file_path, output_folder,known_encodings_list)
                futures[future] = file_path
            for future in concurrent.futures.as_completed(futures):
                file_path = futures[future]
                process_name = f"FaceRecWorker"
                try:
                    result = future.result()
                    logging.info(f"Processed: {file_path}")
                    if result:
                        logging.info(f"Image {file_path} matches with known face")
                        output_file_path=build_matches_file(file_path,output_folder)
                        if not os.path.exists(output_file_path):
                            shutil.copy(file_path,output_file_path)
                        else:
                            logging.info(f"File {file_path} already exists")
                        with lock:
                            config.matches_found.append(file_path)
                    if config.stop_flag==True:
                        logging.warning("Stop flag detected. Cancelling remaining processing.")
                        for future in futures:
                            future.cancel()
                        return None
                    with lock:
                        config.processed_count += 1
                        config.processed_files.append(file_path)
                except concurrent.futures.process.BrokenProcessPool as e:
                    logging.warning(f"{process_name}: {e}")
                    return None
                except Exception as e:
                    logging.error(f"{process_name}: Error processing {file_path}: {e}")
                    logging.error(traceback.format_exc())
        logging.info("All images processed")
        if config.matches_found:
            messagebox.showinfo("Matches Found", f"{len(config.matches_found)} Matching Images copied to 'matched_faces' folder.")
        else:
            messagebox.showinfo("No Match", "No matching images found!")
        return None
    
    def update_ui(self):
        matches_count=len(config.matches_found)
        if config.matches_found:
            if not self.current_image == config.matches_found[matches_count-1]:
                self.current_image = config.matches_found[matches_count-1]
                self.display_image(self.current_image, self.match_image_label)
        if config.processed_count > 0:
            progress_ratio=config.processed_count / (config.total_images or 1)
            self.progress_bar.set(progress_ratio)
            self.label.configure(text=f"{matches_count} images matches, {config.processed_count} / {config.total_images} processed ")
        self.after(2000, self.update_ui)

    def queue_only_root_folder(self):
        if not self.root_folder:
            messagebox.showerror("Error", "Please select a root folder first!")
            return
        logging.info("Queueing only images in root folder")
        self.queue_images(folders=[self.root_folder])

def build_matches_file(file_path,output_folder):
    parent_folder = os.path.basename(os.path.dirname(file_path))
    file_name = os.path.basename(file_path)
    new_file_name = f"{parent_folder}_{file_name}"
    return os.path.join(output_folder, new_file_name)

def load_known_encodings_from_db():
    known_encodings = []
    known_person_ids = []
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        # Get all persons and their known images
        c.execute("""
            SELECT persons.id, image_encodings.encoding
            FROM known_images
            JOIN persons ON known_images.person_id = persons.id
            JOIN image_encodings ON known_images.image_id = image_encodings.image_id
        """)
        rows = c.fetchall()
        for person_id, encoding_blob in rows:
            encoding = np.frombuffer(encoding_blob, dtype=np.float64)
            known_encodings.append(encoding)
            known_person_ids.append(person_id)
    except Exception as e:
        logging.error(f"SQLite error in load_known_encodings_from_db: {e}")
    finally:
        conn.close()
    if not known_encodings:
        logging.error("No valid face encodings found in known_images/image_encodings tables!")
    logging.error(f"Known encodings: {known_encodings}, Known person IDs: {known_person_ids}")
    return known_encodings, known_person_ids

def process_image(file_path,output_folder, known_encodings_list):
    try:
        output_file_path=build_matches_file(file_path,output_folder)
        if os.path.exists(output_file_path):
            return file_path
        unknown_image = face_recognition.load_image_file(file_path)
        unknown_encodings = face_recognition.face_encodings(unknown_image)
        known_encodings = [np.array(enc) for enc in known_encodings_list]
        for idx, encoding in enumerate(unknown_encodings):
            matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.5)
            for i, is_match in enumerate(matches):
                if is_match:
                    person_id = known_person_ids[i]
                    insert_match(str(file_path), person_id)
                    return file_path
    except Exception as e:
        print(f"Issue Processing {file_path}: {e}")
    return None

def worker_init(ppid):
    process_name = f"face_worker_{os.getpid()}"
    setproctitle.setproctitle(process_name)
    pid = os.getpid()

if __name__ == "__main__":
    init_db()
    config.initialize()
    utils.init_logging() 
    logging.info("Face Recognition App Starts")
    logging.info(f"Your computer have {os.cpu_count()} CPUs, configure workers accordingly")
    app = FaceRecognitionApp()
    app.mainloop()
    logging.info("Face Recognition App Exits")