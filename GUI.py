# # import customtkinter as ctk
# # from customtkinter import filedialog
# # import numpy as np
# # import tensorflow as tf
# # from PIL import Image, ImageTk
# # from keras.applications.resnet50 import ResNet50
# # from keras.preprocessing import image
# # import matplotlib.pyplot as plt

# # class ModelGUI:
# #     def __init__(self, root):
# #         self.root = root
# #         self.root.title("Model Evaluation GUI")
# #         self.root.geometry("800x600")

# #         self.model = None

# #         self.load_model_button = ctk.CTkButton(root, text="Load Model", command=self.load_model)
# #         self.load_model_button.pack(pady=10)

# #         self.image_label = ctk.CTkLabel(root)
# #         self.image_label.pack(pady=10)

# #         self.classify_button = ctk.CTkButton(root, text="Classify Image", command=self.classify_image)
# #         self.classify_button.pack(pady=5)
# #         self.classify_button.configure(state="disabled")

# #         self.result_label = ctk.CTkLabel(root, text="")
# #         self.result_label.pack(pady=10)

# #         self.vgg_model = ResNet50(weights='imagenet')

# #     def load_model(self):
# #         model_path = filedialog.askopenfilename(filetypes=[("HDF5 files", "*.h5")])
# #         if model_path:
# #             try:
# #                 self.model = tf.keras.models.load_model(model_path)
# #                 self.classify_button.configure(state="normal")
# #                 self.show_info_message("Success", "Model loaded successfully!")
# #             except Exception as e:
# #                 self.show_error_message("Error", f"Failed to load model: {str(e)}")

# #     def classify_image(self):
# #         if self.model is None:
# #             self.show_warning_message("Warning", "Please load a model first!")
# #             return

# #         image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
# #         if image_path:
# #             try:
# #                 img = Image.open(image_path)
# #                 img = img.resize((160, 160))  # Resize image to match model input size
# #                 img_array = image.img_to_array(img)
# #                 img_array = np.expand_dims(img_array, axis=0)
# #                 img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

# #                 prediction = self.model.predict(img_array)
# #                 class_label = "Malignant" if prediction[0][0] > 0.5 else "Benign"  # Example classification logic
# #                 confidence = prediction[0][0]

# #                 self.result_label.configure(text=f"Predicted class: {class_label}\nConfidence: {confidence:.2f}")

# #                 img = Image.open(image_path)
# #                 img.thumbnail((400, 400))  # Resize image for display

# #                 # Convert PIL.Image to PhotoImage
# #                 photo_image = ImageTk.PhotoImage(img)
# #                 self.image_label.configure(image=photo_image)
# #                 self.image_label.image = photo_image

# #                 # Plot prediction probabilities
# #                 plt.figure()
# #                 plt.bar(["Benign", "Malignant"], [1 - confidence, confidence])
# #                 plt.title("Prediction Probabilities")
# #                 plt.xlabel("Class")
# #                 plt.ylabel("Probability")
# #                 plt.show()

# #             except Exception as e:
# #                 self.show_error_message("Error", f"Failed to classify image: {str(e)}")

# #     def show_info_message(self, title, message):
# #         # Show info message using CTkLabel
# #         self.result_label.configure(text=message)

# #     def show_warning_message(self, title, message):
# #         # Show warning message using CTkLabel
# #         self.result_label.configure(text=message)

# #     def show_error_message(self, title, message):
# #         # Show error message using CTkLabel
# #         self.result_label.configure(text=message)

# # if __name__ == "__main__":
# #     root = ctk.CTk()
# #     app = ModelGUI(root)
# #     root.mainloop()


# import customtkinter as ctk
# from customtkinter import filedialog
# import numpy as np
# import tensorflow as tf
# from PIL import Image, ImageTk
# from keras.applications.resnet50 import ResNet50
# from keras.preprocessing import image
# import matplotlib.pyplot as plt
# from sklearn.metrics import classification_report

# class ModelGUI:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("Model Evaluation GUI")
#         self.root.geometry("800x600")

#         self.model = None

#         self.load_model_button = ctk.CTkButton(root, text="Load Model", command=self.load_model)
#         self.load_model_button.pack(pady=10)

#         self.image_label = ctk.CTkLabel(root)
#         self.image_label.pack(pady=10)

#         self.classify_button = ctk.CTkButton(root, text="Classify Image", command=self.classify_image)
#         self.classify_button.pack(pady=5)
#         self.classify_button.configure(state="disabled")

#         self.result_label = ctk.CTkLabel(root, text="")
#         self.result_label.pack(pady=10)

#         self.vgg_model = ResNet50(weights='imagenet')

#     def load_model(self):
#         model_path = filedialog.askopenfilename(filetypes=[("HDF5 files", "*.h5")])
#         if model_path:
#             try:
#                 self.model = tf.keras.models.load_model(model_path)
#                 self.classify_button.configure(state="normal")
#                 self.show_info_message("Success", "Model loaded successfully!")
#             except Exception as e:
#                 self.show_error_message("Error", f"Failed to load model: {str(e)}")

#     def classify_image(self):
#         if self.model is None:
#             self.show_warning_message("Warning", "Please load a model first!")
#             return

#         image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
#         if image_path:
#             try:
#                 img = Image.open(image_path)
#                 img = img.resize((160, 160))  # Resize image to match model input size
#                 img_array = image.img_to_array(img)
#                 img_array = np.expand_dims(img_array, axis=0)
#                 img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

#                 prediction = self.model.predict(img_array)
#                 class_label = "Malignant" if prediction[0][0] > 0.5 else "Benign"  # Example classification logic
#                 confidence = prediction[0][0]

#                 # Evaluate the model and get evaluation report
#                 test_data = [img]  # Assuming single image for testing
#                 test_labels = [0]  # Example label, replace with actual label
#                 accuracy, report = self.evaluate_model(test_data, test_labels)

#                 # Update result label text with prediction, confidence, and accuracy
#                 result_text = f"Predicted class: {class_label}\n"
#                 result_text += f"Confidence: {confidence:.2f}\n"
#                 result_text += f"Accuracy: {accuracy:.2f}\n\n"
#                 for key, value in report.items():
#                     if isinstance(value, dict):
#                         result_text += f"{key}:\n"
#                         for metric, metric_value in value.items():
#                             result_text += f"  - {metric}: {metric_value:.2f}\n"
#                     else:
#                         result_text += f"{key}: {value:.2f}\n"

#                 self.result_label.configure(text=result_text)

#                 # Display image in GUI
#                 img = Image.open(image_path)
#                 img.thumbnail((400, 400))  # Resize image for display
#                 photo_image = ImageTk.PhotoImage(img)
#                 self.image_label.configure(image=photo_image)
#                 self.image_label.image = photo_image

#                 # Plot prediction probabilities
#                 plt.figure()
#                 plt.bar(["Benign", "Malignant"], [1 - confidence, confidence])
#                 plt.title("Prediction Probabilities")
#                 plt.xlabel("Class")
#                 plt.ylabel("Probability")
#                 plt.show()

#             except Exception as e:
#                 self.show_error_message("Error", f"Failed to classify image: {str(e)}")


#     def evaluate_model(self, test_data, test_labels):
#         # Preprocess the test data
#         test_data = np.array([image.img_to_array(img.resize((160, 160))) for img in test_data])
#         test_data = tf.keras.applications.resnet50.preprocess_input(test_data)

#         # Make predictions
#         predictions = self.model.predict(test_data)
#         predicted_classes = (predictions > 0.5).astype(int)

#         # Calculate evaluation metrics
#         accuracy = np.mean(predicted_classes == test_labels)
#         report = classification_report(test_labels, predicted_classes, output_dict=True)

#         return accuracy, report



#     def show_info_message(self, title, message):
#         # Show info message using CTkLabel
#         self.result_label.configure(text=message)

#     def show_warning_message(self, title, message):
#         # Show warning message using CTkLabel
#         self.result_label.configure(text=message)

#     def show_error_message(self, title, message):
#         # Show error message using CTkLabel
#         self.result_label.configure(text=message)

#     def show_evaluation_report(self, test_data, test_labels):
#         if self.model is None:
#             self.show_warning_message("Warning", "Please load a model first!")
#             return

#         try:
#             # Preprocess the test data
#             test_data = np.array([image.img_to_array(img.resize((160, 160))) for img in test_data])
#             test_data = tf.keras.applications.resnet50.preprocess_input(test_data)

#             # Make predictions
#             predictions = self.model.predict(test_data)
#             predicted_classes = (predictions > 0.5).astype(int)

#             # Calculate evaluation metrics
#             accuracy = np.mean(predicted_classes == test_labels)
#             report = classification_report(test_labels, predicted_classes, output_dict=True)

#             # Create a new window for the report
#             report_window = ctk.CTk()
#             report_window.title("Model Evaluation Report")

#             # Display accuracy in the report
#             accuracy_label = ctk.CTkLabel(report_window, text=f"Accuracy: {accuracy:.2f}")
#             accuracy_label.pack()

#             # Display other evaluation metrics
#             for key, value in report.items():
#                 if isinstance(value, dict):
#                     label = ctk.CTkLabel(report_window, text=f"{key}:")
#                     label.pack()
#                     for metric, metric_value in value.items():
#                         metric_label = ctk.CTkLabel(report_window, text=f"  - {metric}: {metric_value:.2f}")
#                         metric_label.pack()
#                 else:
#                     label = ctk.CTkLabel(report_window, text=f"{key}: {value:.2f}")
#                     label.pack()

#             # Pack the report window
#             report_window.mainloop()

#         except Exception as e:
#             self.show_error_message("Error", f"Failed to generate evaluation report: {str(e)}")

# if __name__ == "__main__":
#     root = ctk.CTk()
#     app = ModelGUI(root)
#     root.mainloop()




import customtkinter as ctk
from customtkinter import filedialog
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk
from sklearn.metrics import classification_report

class ModelGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Model Evaluation GUI")
        self.root.geometry("800x600")

        self.model = None
        self.image = None
        self.prediction = None

        self.load_model_button = ctk.CTkButton(root, text="Load Model", command=self.load_model)
        self.load_model_button.pack(pady=10)

        self.image_label = ctk.CTkLabel(root)
        self.image_label.pack(pady=10)

        self.classify_button = ctk.CTkButton(root, text="Classify Image", command=self.classify_image)
        self.classify_button.pack(pady=5)
        self.classify_button.configure(state="disabled")

        self.result_label = ctk.CTkLabel(root, text="")
        self.result_label.pack(pady=10)

    def load_model(self):
        model_path = filedialog.askopenfilename(filetypes=[("HDF5 files", "*.h5")])
        if model_path:
            try:
                self.model = tf.keras.models.load_model(model_path)
                self.classify_button.configure(state="normal")
                self.show_info_message("Success", "Model loaded successfully!")
            except Exception as e:
                self.show_error_message("Error", f"Failed to load model: {str(e)}")

    def classify_image(self):
        if self.model is None:
            self.show_warning_message("Warning", "Please load a model first!")
            return

        image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if image_path:
            try:
                # Load and preprocess the image
                self.image = Image.open(image_path)
                self.image = self.image.resize((160, 160))  # Resize image to match model input size
                img_array = tf.keras.preprocessing.image.img_to_array(self.image)
                img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
                img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

                # Make a prediction for a single image
                self.prediction = self.model.predict(img_array)
                class_label = "Malignant" if self.prediction[0][0] > 0.5 else "Benign"
                confidence = self.prediction[0][0] if class_label == "Malignant" else 1 - self.prediction[0][0]

                self.result_label.configure(text=f"Predicted class: {class_label}\nConfidence: {confidence:.2f}")

                # Display image in GUI
                photo_image = ImageTk.PhotoImage(self.image)
                self.image_label.configure(image=photo_image)
                self.image_label.image = photo_image

                # Show detailed report
                self.show_detailed_report()

            except Exception as e:
                self.show_error_message("Error", f"Failed to classify image: {str(e)}")


    def show_detailed_report(self):
        if self.image is None:
            self.show_warning_message("Warning", "Please classify an image first!")
            return

        try:
            # Preprocess the image
            img_array = tf.keras.preprocessing.image.img_to_array(self.image)
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

            # Make a prediction for a single image
            prediction = self.model.predict(img_array)
            predicted_probability = prediction[0]
            predicted_class = 'Malignant' if predicted_probability > 0.5 else 'Benign'

            # Assuming y_true is a list of true labels, e.g., [0, 1, 0, 1, ...]
            # Replace the following line with the actual true labels for your predictions
            y_true = [0]  # Example true label

            # Convert the prediction probability to a binary format for the report
            y_pred = [1 if predicted_probability > 0.5 else 0]

            # Define the target names for binary classification
            target_names = ['Benign', 'Malignant']

            # Generate the classification report
            report = classification_report(y_true, y_pred, labels=[0, 1], target_names=target_names, output_dict=True)

            # Create a new window for the detailed report
            report_window = ctk.CTkToplevel(self.root)
            report_window.title("Detailed Report")
            report_window.geometry("600x400")

            # Center the window on the screen
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            window_width = report_window.winfo_reqwidth()
            window_height = report_window.winfo_reqheight()
            position_x = int((screen_width / 2) - (window_width / 2))
            position_y = int((screen_height / 2) - (window_height / 2))
            report_window.geometry(f"+{position_x}+{position_y}")

            # Create a text widget to display the report
            report_text = ctk.CTkTextbox(report_window)
            report_text.pack(padx=10, pady=10, fill="both", expand=True)

            # Populate the text widget with the classification report
            report_text.insert("0.0", "Classification Report:\n")
            for key, value in report.items():
                if isinstance(value, dict):
                    report_text.insert("end", f"{key}:\n")
                    for metric, metric_value in value.items():
                        report_text.insert("end", f"  - {metric}: {metric_value:.2f}\n")
                else:
                    report_text.insert("end", f"{key}: {value:.2f}\n")

        except Exception as e:
            self.show_error_message("Error", f"Failed to generate detailed report: {str(e)}")



    def show_info_message(self, title, message):
        self.show_message_box(title, message, "info")

    def show_warning_message(self, title, message):
        self.show_message_box(title, message, "warning")

    def show_error_message(self, title, message):
        self.show_message_box(title, message, "error")

    def show_message_box(self, title, message, message_type):
        message_box = ctk.CTkToplevel(self.root)
        message_box.title(title)
        message_box.geometry("300x100")

        # Center the window on the screen
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = message_box.winfo_reqwidth()
        window_height = message_box.winfo_reqheight()
        position_x = int((screen_width / 2) - (window_width / 2))
        position_y = int((screen_height / 2) - (window_height / 2))
        message_box.geometry(f"+{position_x}+{position_y}")

        message_label = ctk.CTkLabel(message_box, text=message)
        message_label.pack(padx=10, pady=10)

        ok_button = ctk.CTkButton(message_box, text="OK", command=message_box.destroy)
        ok_button.pack(pady=10)

if __name__ == "__main__":
    root = ctk.CTk()
    app = ModelGUI(root)
    root.mainloop()