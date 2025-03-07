import os,shutil
import face_recognition
import customtkinter as ctk
import logging
import threading
import utils,config
from tkinter import filedialog, messagebox,simpledialog
from PIL import Image
from pathlib import Path

# Initialize the app
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class FaceRecognitionApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Face Recognition App")
        width = 600
        height = 680

        # Get the screen dimensions
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        # Calculate the position to center the window
        x = (screen_width / 2) - (width / 2)
        y = (screen_height / 2) - (height / 2)

        # Set the geometry of the window
        self.geometry(f'{width}x{height}+{int(x)}+{int(y)}')
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
        
        self.exit_button=ctk.CTkButton(self, text="Exit Application", fg_color="red", hover_color="darkred",command=self.exit_app)
        self.exit_button.pack(pady=10)
        
        self.image_label = ctk.CTkLabel(self, text="")
        self.image_label.pack(pady=10)
        
        self.match_image_label = ctk.CTkLabel(self, text="")
        self.match_image_label.pack(pady=10)
             
        self.selected_image_path = None
        self.root_folder = None
        self.matching_folders = []

    
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
        total_images = 0
        for afolder in self.matching_folders:
            try:
                # Get all files in the folder
                folder=Path(afolder)
                files = os.listdir(folder)
                logging.debug(f"Files in {folder}: {files}")  # Log the files in each folder
                # Filter files with the correct image extensions
                image_files = [f for f in files if f.lower().endswith(('jpg', 'jpeg', 'png')) and os.path.isfile(os.path.join(folder, f))]
                logging.debug(f"Image files in {folder}: {image_files}")  # Log only the image files found
                # Add the count of image files found in this folder to the total
                total_images += len(image_files)
            except Exception as e:
                logging.warning(f"Error accessing folder '{folder}': {e}")
                continue   
        return total_images
    
    def start_comparison(self):
        thread = threading.Thread(target=self.compare_faces)
        thread.start()
    
    def compare_faces(self):
        if not self.selected_image_path or not self.matching_folders:
            messagebox.showerror("Error", "Please select an image and a root folder with matching subfolders!")
            return
        
        output_folder = config.OUTPUT
        os.makedirs(output_folder, exist_ok=True)
        logging.info(f"Load known face")
        known_image = face_recognition.load_image_file(self.selected_image_path)
        known_encoding = face_recognition.face_encodings(known_image)
        
        if not known_encoding:
            messagebox.showerror("Error", "No face found in selected image!")
            return
        
        known_encoding = known_encoding[0]
        matches_found = []
        matches_count = 0
        processed_count = 0
        total_images = self.count_images_in_folders()
        
        logging.info(f"{total_images} images to process")
        
        for afolder in self.matching_folders:
            folder=Path(afolder)
            logging.info(f"Processing images in folder {folder}")
            image_files = [f for f in os.listdir(folder) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
            for filename in image_files:
                file_path = os.path.join(folder, filename)

                logging.info(f"Processing {filename}")
                processed_count +=1
                try:
                    unknown_image = face_recognition.load_image_file(file_path)
                    unknown_encodings = face_recognition.face_encodings(unknown_image)
                    
                    for encoding in unknown_encodings:
                        match = face_recognition.compare_faces([known_encoding], encoding, tolerance=0.6)
                        if match[0]: 
                            matches_found.append(filename)
                            logging.info(f"Image {filename} matches with known face")
                            matches_count +=1
                            self.display_image(file_path, self.match_image_label)
                            shutil.copy(file_path, os.path.join(output_folder, filename))
                            break
                except Exception as e:
                    logging.warning(f"Skipping {filename}: {e}")
                    continue
                
                # Update progress bar
                self.progress_bar.set(processed_count / total_images)
                self.update_idletasks()
        
        if matches_found:
            messagebox.showinfo("Matches Found", f"{matches_count} Matching Images copied to 'matched_faces' folder: \n")      
        else:
            messagebox.showinfo("No Match", "No matching images found!")

if __name__ == "__main__":
    config.initialize()
    utils.init_logging() 
    logging.info(f"Face Recognition App Starts")
    app = FaceRecognitionApp()
    app.mainloop()
    logging.info(f"Face Recognition App Ended")