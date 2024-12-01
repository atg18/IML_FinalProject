import tkinter as tk
from tkinter import messagebox, filedialog
import cv2
import os
import numpy as np
from datetime import datetime
import csv

# Load pre-trained face detection and recognition model
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_trainer.yml")

# Load names from training
names = {}
dataset_path = "face_dataset"
for label, person_name in enumerate(os.listdir(dataset_path)):
    names[label] = person_name

# Attendance log file
attendance_file = "attendance_log.csv"

# Create the attendance log file if it doesn't exist
if not os.path.exists(attendance_file):
    with open(attendance_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Timestamp"])

# Real-time face recognition
def start_recognition():
    cap = cv2.VideoCapture(0)
    attendance = {}
    messagebox.showinfo("Face Recognition", "Starting webcam. Press 'ESC' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Error", "Failed to access the camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            label, confidence = recognizer.predict(face)

            name = names[label] if confidence < 100 else "Unknown"
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if name != "Unknown" and name not in attendance:
                attendance[name] = timestamp
                with open(attendance_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([name, timestamp])

            # Draw rectangle and name
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow("Face Recognition", frame)

        # Exit on 'ESC'
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Attendance Logged", "Attendance has been logged successfully.")

# Browse dataset path for training
def select_dataset():
    dataset_dir = filedialog.askdirectory(title="Select Face Dataset Directory")
    if dataset_dir:
        global dataset_path
        dataset_path = dataset_dir
        messagebox.showinfo("Dataset Selected", f"Dataset path set to: {dataset_dir}")

# Train the model
def train_model():
    faces = []
    labels = []
    label = 0

    if not os.path.exists(dataset_path):
        messagebox.showerror("Error", "Dataset path does not exist. Please select a valid dataset path.")
        return

    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)

        if os.path.isdir(person_path):
            for image_name in os.listdir(person_path):
                if image_name.endswith(".jpg"):
                    image_path = os.path.join(person_path, image_name)
                    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    faces_detected = face_detector.detectMultiScale(gray_image, 1.3, 5)

                    for (x, y, w, h) in faces_detected:
                        face = gray_image[y:y + h, x:x + w]
                        faces.append(face)
                        labels.append(label)
            label += 1

    if faces:
        recognizer.train(faces, np.array(labels))
        recognizer.save("face_trainer.yml")
        messagebox.showinfo("Training Complete", "Model trained and saved as 'face_trainer.yml'.")
    else:
        messagebox.showerror("Error", "No faces detected in the dataset.")

# GUI setup
root = tk.Tk()
root.title("Face Detection Attendance System")
root.geometry("500x300")

title_label = tk.Label(root, text="Face Detection Attendance System", font=("Helvetica", 16))
title_label.pack(pady=10)

dataset_button = tk.Button(root, text="Select Dataset", command=select_dataset, font=("Helvetica", 12))
dataset_button.pack(pady=10)

train_button = tk.Button(root, text="Train Model", command=train_model, font=("Helvetica", 12))
train_button.pack(pady=10)

recognition_button = tk.Button(root, text="Start Recognition", command=start_recognition, font=("Helvetica", 12))
recognition_button.pack(pady=10)

exit_button = tk.Button(root, text="Exit", command=root.quit, font=("Helvetica", 12))
exit_button.pack(pady=10)

root.mainloop()
