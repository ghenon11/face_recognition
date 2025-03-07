import os
import shutil
import logging, traceback
import threading
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
        self.geometry("600x680")
        self.protocol("WM_DELETE_WINDOW", self.confirm_close)

        self.label = ctk.CTkLabel(self, text="Select an Image to Compare")
        self.label.pack(pady=10)

        self.select_button = ctk.CTkButton(self, text="Select Image", command=self.select_image)
        self.select_button.pack(pady=10)

        self.folder_button = ctk.CTkButton(self, text="Select Root Folder", command=self.select_root_folder)
        self.folder_button.pack(pady=10)

        self.compare_button = ctk.CTkButton(self, text="Compare Faces", command=self.start_comparison, state="disabled")
        self.compare_button.pack(pady=10)

        self.progress_bar = ctk.CTkProgressBar(self)
        self.progress_bar.pack(pady=10)
        self.progress_bar.set(0)

        self.exit_button = ctk.CTkButton(self, text="Exit Application", fg_color="red", hover_color="darkred", command=self.exit_app)
        self.exit_button.pack(pady=10)

        self.image_label = ctk.CTkLabel(self, text="")
        self.image_label.pack(pady=10)

        self.match_image_label = ctk.CTkLabel(self, text="")
        self.match_image_label.pack(pady=10)

        self.selected_image_path = None
        self.root_folder = None
        self.matching_folders = []
        self.image_queue = Queue()
        self.current_image = None
        
        self.update_ui()

    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            self.selected_image_path = file_path
            self.label.configure(text=f"Selected: {os.path.basename(file_path)}")
            self.display_image(file_path, self.image_label)
            if self.root_folder:
                self.compare_button.configure(state="normal")

    def display_image(self, file_path, label):
        image = Image.open(file_path)
        aspect_ratio = image.width / image.height
        new_width = 200
        new_height = int(new_width / aspect_ratio)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        photo = ctk.CTkImage(light_image=image, dark_image=image, size=(new_width, new_height))
        label.configure(image=photo)
        label.image = photo

    def select_root_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            search_string = simpledialog.askstring("Input", "Enter string to match subfolders:")
            if search_string:
                self.root_folder = folder
                self.matching_folders = [os.path.join(folder, sub) for sub in os.listdir(folder) if search_string in sub and os.path.isdir(os.path.join(folder, sub))]
                self.label.configure(text=f"Matching Folders: {len(self.matching_folders)} found")
                if self.selected_image_path:
                    self.compare_button.configure(state="normal")

    def confirm_close(self):
        if messagebox.askyesno("Confirm Close", "Are you sure you want to close?"):
            self.exit_app()

    def exit_app(self):
        logging.info("Exiting Application")
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
        """Queue all images for processing."""
        logging.info(f"Building image list")
        for folder in self.matching_folders:
            try:
                image_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
                for img in image_files:
                    formatted_path=Path(img)
                    self.image_queue.put(formatted_path)
            except Exception as e:
                logging.warning(f"Skipping folder {folder}: {e}")

    def start_comparison(self):
        thread = threading.Thread(target=self.compare_faces)
        thread.start()
              
    def compare_faces(self):
        """Compare faces using multi-threading with ThreadPoolExecutor and a Queue."""
        if not self.selected_image_path or not self.matching_folders:
            messagebox.showerror("Error", "Please select an image and a root folder with matching subfolders!")
            return

        output_folder = config.OUTPUT
        os.makedirs(output_folder, exist_ok=True)

        # Load known face encoding
        logging.info("Loading known face")
        known_image = face_recognition.load_image_file(self.selected_image_path)
        known_encodings = face_recognition.face_encodings(known_image)

        if not known_encodings:
            messagebox.showerror("Error", "No face found in selected image!")
            return

        self.queue_images()  # Populate the queue
        config.total_images = self.image_queue.qsize()
        items = list(self.image_queue.queue)
        items_len=len(items)
        logging.info(f"{items_len} images to process")

        config.processed_count = 0

        with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(process_image, item, known_encodings,config): item for item in items}

            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()  # Get the result of process_image
                    config.processed_count += 1
                    if result:  # If a match was found
                        with lock:
                            config.matches_found.append(file_path)
                            shutil.copy(file_path, os.path.join(output_folder, os.path.basename(file_path)))
                            logging.info(f"Image {file_path} matches with known face")
                    else:
                            logging.info(f"{file_path} Processed")
                except Exception as e:
                    logging.error(f"Error processing image {futures[future]}: {e}")
                    logging.error("Traceback: %s", traceback.format_exc())
                      
        logging.info(f"Processing ended")
        # Show results
        if config.matches_found:
            messagebox.showinfo("Matches Found", f"{len(config.matches_found)} Matching Images copied to 'matched_faces' folder.")
        else:
            messagebox.showinfo("No Match", "No matching images found!")
    
    def update_ui(self):
        if config.matches_found:
            if not self.current_image == config.matches_found[len(config.matches_found)-1]:
                self.current_image = config.matches_found[len(config.matches_found)-1]
                self.display_image(self.current_image, self.match_image_label)
        if config.total_images>0:
            self.progress_bar.set(config.processed_count / config.total_images)
        self.after(2000, self.update_ui)  # Update every 2 seconds

def process_image(file_path, known_encodings_list):
    """Process a single image for face matching."""
    try:
        unknown_image = face_recognition.load_image_file(file_path)
        unknown_encodings = face_recognition.face_encodings(unknown_image)

        for encoding in unknown_encodings:
            encoding_list = encoding.tolist()  # Convert NumPy array to list
            match = face_recognition.compare_faces(known_encodings_list, encoding_list, tolerance=0.6)
            if match[0]:  # If at least one face matches
                return file_path  # Return matched file path

    except Exception as e:
        logging.warning(f"Skipping {file_path}: {e}")

    return None  # No match found

if __name__ == "__main__":
    config.initialize()
    utils.init_logging() 
    logging.info("Face Recognition App Starts")
    app = FaceRecognitionApp()
    app.mainloop()
    logging.info("Face Recognition App Ended")
