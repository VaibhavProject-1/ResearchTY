# import customtkinter as ctk
# from customtkinter import filedialog
# from tkinter import messagebox
# from ultralytics import YOLO
# from PIL import Image, ImageTk
# from tkinter import StringVar

# class YOLOv8_GUI:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("YOLOv8 Inference GUI")
        
#         self.model = None
#         self.image_path = None
        
#         self.create_widgets()
        
#     def create_widgets(self):
#         # Model Selection
#         self.model_label = ctk.CTkLabel(self.root, text="Select YOLOv8 Model:")
#         self.model_label.grid(row=0, column=0, sticky='w', padx=10, pady=10)
        
#         self.model_path = StringVar()
#         self.model_entry = ctk.CTkEntry(self.root, textvariable=self.model_path, width=50)
#         self.model_entry.grid(row=0, column=1, padx=10, pady=10)
        
#         self.browse_model_button = ctk.CTkButton(self.root, text="Browse", command=self.load_model)
#         self.browse_model_button.grid(row=0, column=2, padx=10, pady=10)
        
#         # Image Selection
#         self.image_label = ctk.CTkLabel(self.root, text="Select Image:")
#         self.image_label.grid(row=1, column=0, sticky='w', padx=10, pady=10)
        
#         self.image_path = StringVar()
#         self.image_entry = ctk.CTkEntry(self.root, textvariable=self.image_path, width=50)
#         self.image_entry.grid(row=1, column=1, padx=10, pady=10)
        
#         self.browse_image_button = ctk.CTkButton(self.root, text="Browse", command=self.load_image)
#         self.browse_image_button.grid(row=1, column=2, padx=10, pady=10)
        
#         # Inference Button
#         self.run_button = ctk.CTkButton(self.root, text="Run Inference", command=self.run_inference)
#         self.run_button.grid(row=2, column=1, padx=10, pady=10)
        
#         # Result Display
#         self.result_label = ctk.CTkLabel(self.root, text="Inference Results:")
#         self.result_label.grid(row=3, column=0, sticky='w', padx=10, pady=10)
        
#         self.result_text = ctk.CTkLabel(self.root, width=60, height=10, wraplength=500)  # Use CTkLabel for multiline text display
#         self.result_text.grid(row=4, column=0, columnspan=3, padx=10, pady=10)
        
#     def load_model(self):
#         model_path = filedialog.askopenfilename(title="Select YOLOv8 Model")
#         if model_path:
#             self.model_entry.delete(0, ctk.END)
#             self.model_entry.insert(ctk.END, model_path)
#             self.model = YOLO(model_path)
#             messagebox.showinfo("Model Loaded", "Model loaded successfully!")
    
#     def load_image(self):
#         image_path = filedialog.askopenfilename(title="Select Image")
#         if image_path:
#             self.image_entry.delete(0, ctk.END)
#             self.image_entry.insert(ctk.END, image_path)
    
#     def run_inference(self):
#         if not self.model:
#             messagebox.showerror("Error", "Please select a model first!")
#             return
#         if not self.image_entry.get():
#             messagebox.showerror("Error", "Please select an image!")
#             return
        
#         image_path = self.image_entry.get()
#         results = self.model(image_path)
#         self.display_results(results)
    
#     def display_results(self, results):
#         # Clear the text widget
#         self.result_text.configure(text="")
        
#         # Check if the results object is a list
#         if isinstance(results, list):
#             # If it's a list, iterate through each result and display it
#             result_message = ""
#             for idx, result in enumerate(results):
#                 result_message += f"Result {idx + 1}:\n"
#                 result_message += self.construct_single_result_message(result)
#                 result_message += "\n\n"
#             self.result_text.configure(text=result_message)
#         else:
#             # If it's not a list, assume it's a single result and display it
#             result_message = self.construct_single_result_message(results)
#             self.result_text.configure(text=result_message)

#     def construct_single_result_message(self, result):
#         # Extract relevant information from the result object
#         predicted_class = list(result.names.values())[result.probs.top1]
#         top5_classes = [list(result.names.values())[idx] for idx in result.probs.top5]
#         top1_confidence = result.probs.top1conf.item()  # Convert to Python scalar
#         top5_confidences = [conf.item() for conf in result.probs.top5conf]  # Convert to Python scalars
        
#         # Construct the result message
#         result_message = f"Predicted Class: {predicted_class}\n"
#         result_message += f"Top 5 Classes: {top5_classes}\n"
#         result_message += f"Top 1 Confidence: {top1_confidence:.4f}\n"
#         result_message += f"Top 5 Confidences: {top5_confidences}\n"
#         result_message += f"Original Image Path: {result.path}\n"
#         result_message += f"Original Image Shape: {result.orig_shape}\n"
#         result_message += f"Inference Speed (inference stage): {result.speed['inference']:.2f} seconds\n"
        
#         return result_message

# if __name__ == "__main__":
#     root = ctk.CTk()
#     app = YOLOv8_GUI(root)
#     root.mainloop()


import customtkinter as ctk
from customtkinter import filedialog
from tkinter import messagebox
from ultralytics import YOLO
from PIL import Image, ImageTk
from tkinter import StringVar

class YOLOv8_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 Inference GUI")
        
        self.model = None
        self.image_path = None
        self.selected_image = None
        
        self.create_widgets()
        
    def create_widgets(self):
        # Model Selection
        self.model_label = ctk.CTkLabel(self.root, text="Select YOLOv8 Model:")
        self.model_label.grid(row=0, column=0, sticky='w', padx=10, pady=10)
        
        self.model_path = StringVar()
        self.model_entry = ctk.CTkEntry(self.root, textvariable=self.model_path, width=50)
        self.model_entry.grid(row=0, column=1, padx=10, pady=10)
        
        self.browse_model_button = ctk.CTkButton(self.root, text="Browse", command=self.load_model)
        self.browse_model_button.grid(row=0, column=2, padx=10, pady=10)
        
        # Image Selection
        self.image_label = ctk.CTkLabel(self.root, text="Select Image:")
        self.image_label.grid(row=1, column=0, sticky='w', padx=10, pady=10)
        
        self.image_path = StringVar()
        self.image_entry = ctk.CTkEntry(self.root, textvariable=self.image_path, width=50)
        self.image_entry.grid(row=1, column=1, padx=10, pady=10)
        
        self.browse_image_button = ctk.CTkButton(self.root, text="Browse", command=self.load_image)
        self.browse_image_button.grid(row=1, column=2, padx=10, pady=10)
        
        # Display Selected Image
        self.selected_image_label = ctk.CTkLabel(self.root, text="Selected Image:")
        self.selected_image_label.grid(row=2, column=0, sticky='w', padx=10, pady=10)
        
        self.image_display = ctk.CTkLabel(self.root)
        self.image_display.grid(row=3, column=0, columnspan=3, padx=10, pady=10)
        
        # Inference Button
        self.run_button = ctk.CTkButton(self.root, text="Run Inference", command=self.run_inference)
        self.run_button.grid(row=4, column=1, padx=10, pady=10)
        
        # Result Display
        self.result_label = ctk.CTkLabel(self.root, text="Inference Results:")
        self.result_label.grid(row=5, column=0, sticky='w', padx=10, pady=10)
        
        self.result_text = ctk.CTkLabel(self.root, width=60, height=10, wraplength=500)  # Use CTkLabel for multiline text display
        self.result_text.grid(row=6, column=0, columnspan=3, padx=10, pady=10)
        
    def load_model(self):
        model_path = filedialog.askopenfilename(title="Select YOLOv8 Model")
        if model_path:
            self.model_entry.delete(0, ctk.END)
            self.model_entry.insert(ctk.END, model_path)
            self.model = YOLO(model_path)
            messagebox.showinfo("Model Loaded", "Model loaded successfully!")
    
    def load_image(self):
        image_path = filedialog.askopenfilename(title="Select Image")
        if image_path:
            self.image_entry.delete(0, ctk.END)
            self.image_entry.insert(ctk.END, image_path)
            self.image_path = image_path
            self.display_selected_image(image_path)
    
    def display_selected_image(self, image_path):
        img = Image.open(image_path)
        img = img.resize((300, 300))  # Resize the image for display
        photo = ImageTk.PhotoImage(img)
        self.image_display.configure(image=photo)
        self.image_display.image = photo  # Keep a reference to avoid garbage collection
    
    def run_inference(self):
        if not self.model:
            messagebox.showerror("Error", "Please select a model first!")
            return
        if not self.image_path:
            messagebox.showerror("Error", "Please select an image!")
            return
        
        results = self.model(self.image_path)
        self.display_results(results)
    
    def display_results(self, results):
        # Clear the text widget
        self.result_text.configure(text="")
        
        # Check if the results object is a list
        if isinstance(results, list):
            # If it's a list, iterate through each result and display it
            result_message = ""
            for idx, result in enumerate(results):
                result_message += f"Result {idx + 1}:\n"
                result_message += self.construct_single_result_message(result)
                result_message += "\n\n"
            self.result_text.configure(text=result_message)
        else:
            # If it's not a list, assume it's a single result and display it
            result_message = self.construct_single_result_message(results)
            self.result_text.configure(text=result_message)

    def construct_single_result_message(self, result):
        # Extract relevant information from the result object
        predicted_class = list(result.names.values())[result.probs.top1]
        top5_classes = [list(result.names.values())[idx] for idx in result.probs.top5]
        top1_confidence = result.probs.top1conf.item()  # Convert to Python scalar
        top5_confidences = [conf.item() for conf in result.probs.top5conf]  # Convert to Python scalars
        
        # Construct the result message
        result_message = f"Predicted Class: {predicted_class}\n"
        result_message += f"Top 5 Classes: {top5_classes}\n"
        result_message += f"Top 1 Confidence: {top1_confidence:.4f}\n"
        result_message += f"Top 5 Confidences: {top5_confidences}\n"
        result_message += f"Original Image Path: {result.path}\n"
        result_message += f"Original Image Shape: {result.orig_shape}\n"
        result_message += f"Inference Speed (inference stage): {result.speed['inference']:.2f} seconds\n"
        
        return result_message

if __name__ == "__main__":
    root = ctk.CTk()
    app = YOLOv8_GUI(root)
    root.mainloop()
