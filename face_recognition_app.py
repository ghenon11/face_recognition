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

lock = threading.Lock()

class FaceRecognitionApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Face Recognition App")
         # Set the dimensions of the window
        width = 800
        height = 680

        # Get the screen dimensions
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        # Calculate the position to center the window
        x = (screen_width / 2) - (width / 2)
        y = (screen_height / 2) - (height / 2)

        # Set the geometry of the window
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


        self.select_button = ctk.CTkButton(self.main_frame, text="Select Image Path", command=self.select_image)
        self.select_button.grid(row=1, column=0, padx=10, pady=5)
        self.label_image_path = ctk.CTkLabel(self.main_frame)
        self.label_image_path.grid(row=1, column=1, padx=10, pady=5)
        self.img_path = ctk.StringVar(value="Set known images path")
        self.label_image_path.configure(textvariable=self.img_path)

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
        
        self.root_folder = None
        self.matching_folders = []
        self.image_queue = Queue()
        self.current_image = None
        
        # Load previous selections from config.ini
        self.root_folder = config.configfile.get("Settings", "root_folder", fallback="")
        self.output_folder = config.configfile.get("Settings", "output_folder", fallback="")
        self.selected_image_path = config.configfile.get("Settings", "selected_image_path", fallback="")

        if self.selected_image_path:
            self.img_path.set(self.selected_image_path)

        if self.root_folder:
            self.label_root_var.set(self.root_folder)
            
        if self.output_folder:
            self.label_output_var.set(self.output_folder)

        if self.selected_image_path and self.root_folder and self.output_folder:
            self.label.configure(text=f"Configuration loaded")
            self.compare_button.configure(state="normal")

        self.update_random_face()
        self.update_ui()

    def update_random_face(self):
        """Pick a random image from the 'faces' directory and display it every 10 seconds."""
        faces_dir = self.selected_image_path
        
        if not os.path.exists(faces_dir):
            logging.warning("Faces directory not found!")
            return

        # Get a list of all image files in the faces directory
        image_files = [os.path.join(faces_dir, f) for f in os.listdir(faces_dir) if f.lower().endswith(('jpg', 'jpeg', 'png'))]

        if not image_files:
            logging.warning("No images found in faces directory!")
            return

        # Pick a random image
        random_image = Path(random.choice(image_files))
        self.display_image(random_image, self.image_label)
        self.after(10000, self.update_random_face) 
        
    def select_image(self):
        folder = filedialog.askdirectory()
        if folder:
            self.selected_image_path = folder
            self.label.configure(text=f"Selected: {os.path.basename(folder)}")
            self.img_path.set(self.selected_image_path)
            config.configfile.set("Settings", "selected_image_path", folder)  # Save to config
            config.save_config()  # Persist changes
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
            search_string = simpledialog.askstring("Input", "Enter string to match subfolders:")
            if search_string:
                self.root_folder = folder
                self.label_root_var.set(self.root_folder)
                config.configfile.set("Settings", "root_folder", folder)  # Save to config
                config.save_config()  # Persist changes
                self.matching_folders = [os.path.join(folder, sub) for sub in os.listdir(folder) if search_string in sub and os.path.isdir(os.path.join(folder, sub))]
                self.label.configure(text=f"Matching Folders: {len(self.matching_folders)} found")
                if self.selected_image_path:
                    self.compare_button.configure(state="normal")
    
    def select_output_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.output_folder = folder
            self.label_output_var.set(self.output_folder)
            config.configfile.set("Settings", "output_folder", folder)  # Save to config
            config.save_config()  # Persist changes
                

    def confirm_close(self):
        if messagebox.askyesno("Confirm Close", "Are you sure you want to close?"):
            self.exit_app()

    def exit_app(self):
        logging.info("Exiting Application")
        config.stop_flag=True
        time.sleep(10)
        self.quit()

    def count_images_in_folders(self):
        """Count total images in the selected folders."""
        total_images = 0
        for folder in self.matching_folders:
            try:
                image_files = [f for f in os.listdir(folder) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
                total_images += len(image_files)
            except Exception as e:
                logging.warning(f"Error accessing folder '{folder}': {e}")
                continue
        return total_images

    def queue_images(self):
        """Queue all images for processing, saving to and loading from a file if available."""
        logging.info("Building image list")
        
        self.queue_file = config.QUEUE_FILE  # Store as an instance variable for periodic updates

        # Check if the queue file exists and is not empty
        if os.path.exists(self.queue_file) and os.path.getsize(self.queue_file) > 0:
            logging.info(f"Loading image queue from {self.queue_file}")
            with open(self.queue_file, "r") as file:
                for line in file:
                    image_path = line.strip()
                    if os.path.exists(image_path):  # Ensure the file still exists
                        self.image_queue.put(Path(image_path))
            logging.info(f"Loaded {self.image_queue.qsize()} images from file.")
        else:
            # If no existing queue file, generate a new queue
            queued_images = []
            if not self.matching_folders:
                messagebox.showerror("Error", "Please select a root folder with matching subfolders!")
                return
            with open(self.queue_file, "w") as file:
                for folder in self.matching_folders:
                    try:
                        image_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
                        for img in image_files:
                            formatted_path = Path(img)
                            self.image_queue.put(formatted_path)  # Enqueue image
                            queued_images.append(str(formatted_path))  # Save as string for writing
                            file.write(str(formatted_path) + "\n")  # Write to file
                    except Exception as e:
                        logging.warning(f"Skipping folder {folder}: {e}")

            logging.info(f"Queued {len(queued_images)} images for processing. Saved to {self.queue_file}")

        # Start periodic saving of the queue
        self.save_queue_periodically()

    def save_queue_periodically(self):
        """Save the remaining queue content to the file every 5 minutes."""
        list_file=[]  # Store as an instance variable for periodic updates

        # Check if the queue file exists and is not empty
        if os.path.exists(self.queue_file) and os.path.getsize(self.queue_file) > 0:
            logging.info(f"Loading image queue from {self.queue_file}")
            with open(self.queue_file, "r") as file:
                for line in file:
                    image_path = line.strip()
                    if os.path.exists(image_path):  # Ensure the file still exists
                        list_file.append(Path(image_path))
       
        with lock:  # Ensure thread safety
            remaining_items = [x for x in list_file if x not in config.processed_files]  
            with open(self.queue_file, "w") as file:
                for item in remaining_items:
                    file.write(str(item) + "\n")  # Save remaining items

        logging.info(f"Saved {len(remaining_items)} remaining items to {self.queue_file}")
        self.after(300000, self.save_queue_periodically)  # Run again in 5 minutes (300,000 ms)




    def start_comparison(self):
        thread = threading.Thread(target=self.compare_faces)
        thread.start()
              
    def compare_faces(self):
        """Compare faces using multiprocessing with detailed logging for each processed file."""
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
        self.queue_images()  # Populate the queue
        config.total_images = self.image_queue.qsize()

        logging.info(f"{config.total_images} images to process")
        config.processed_count = 0
        logging.info(f"Encoding known faces")
        known_encodings = load_known_encodings(self.selected_image_path)
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
                    
                    # **LOG EVERY FILE PROCESSED**
                    logging.info(f"Processed: {file_path}")
                    
                    if result:
                        logging.info(f"Image {file_path} matches with known face")
                        # Copy file with new name
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
        os.remove(config.QUEUE_FILE)
        # Show results
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
        
        self.after(2000, self.update_ui)  # Update every 2 seconds

def build_matches_file(file_path,output_folder):
    parent_folder = os.path.basename(os.path.dirname(file_path))
    file_name = os.path.basename(file_path)
    # New file name format: parent_folder_original_filename.ext
    new_file_name = f"{parent_folder}_{file_name}"
    return os.path.join(output_folder, new_file_name)

def load_known_encodings(faces_dir):
    """Load all known face encodings from the 'faces' directory."""
    known_encodings = []

    if not os.path.exists(faces_dir):
        logging.error("Faces directory does not exist!")
        return known_encodings

    for file_name in os.listdir(faces_dir):
        if file_name.lower().endswith(('jpg', 'jpeg', 'png')):
            file_path = os.path.join(faces_dir, file_name)
            image = face_recognition.load_image_file(file_path)
            encodings = face_recognition.face_encodings(image)

            if encodings:  # Ensure at least one face is detected
                known_encodings.append(encodings[0])
            else:
                logging.warning(f"No face found in {file_name}")

    if not known_encodings:
        logging.error("No valid face encodings found in 'faces' directory!")

    return known_encodings

def process_image(file_path,output_folder, known_encodings_list):
    """Process a single image for face matching."""
    try:
        output_file_path=build_matches_file(file_path,output_folder)
        if os.path.exists(output_file_path): #no  need to check, a previous run saved it !
            return file_path
        unknown_image = face_recognition.load_image_file(file_path)
        unknown_encodings = face_recognition.face_encodings(unknown_image)

        # Convert back to NumPy arrays before comparison
        known_encodings = [np.array(enc) for enc in known_encodings_list]

        for encoding in unknown_encodings:
            match = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.5)
            if any(match):  # If at least one face matches
                return file_path  # Return matched file path

    except Exception as e:
        print(f"Issue Processing {file_path}: {e}")

    return None  # No match found
    
def worker_init(ppid):
    """Initialize worker process with custom name and shared stop flag."""
    
    process_name = f"face_worker_{os.getpid()}"
    setproctitle.setproctitle(process_name)  # Set process name
    pid = os.getpid()
    
if __name__ == "__main__":
    config.initialize()
    utils.init_logging() 
    logging.info("Face Recognition App Starts")
    logging.info(f"Your computer have {os.cpu_count()} CPUs, configure workers accordingly")
    app = FaceRecognitionApp()
    app.mainloop()
    logging.info("Face Recognition App Ended")
