**ATTENDANCE TRACKING USING FACE RECOGNITION**

**1. StoringFaceandName.py**
Purpose: To capture face images of individual users and augment these images to creat a robust dataset for training the model.

**Input Collection:**
• The user enters their name via a Flask-based web interface. This name is used to create a unique folder in the dataset directory for storing images.
Face Capture:
• The system accesses the webcam to capture video frames.
• A Haar Cascade Classifier is used to detect faces in real-time from each video frame.
• Detected faces are cropped, converted to grayscale, and processed for augmentation.

**Image Augmentation:**
The captured face images undergo various transformations to enhance the dataset:
• Rotation: Faces are rotated at multiple angles (e.g., -15° and +15°).
• Scaling: Faces are resized by scaling factors (e.g., 0.9x and 1.1x).
• Flipping: Horizontal flips are applied to reverse the image.
• Translation: Faces are shifted in small steps along the x and y axes (e.g., ±10 pixels).
These augmentations simulate variations in real-world conditions like head tilt, movement, and different perspectives.

**Storage:**
• Each augmented image is saved in a folder named after the user within the face dataset directory. The filenames include metadata about the augmentation type.
  Output: A dataset of augmented face images stored for each user, organized by individual directories

**2. TrainingModel.py**
Purpose: To train a facial recognition model using the augmented face dataset created during the registration stage.

Dataset Loading:
• The system reads images from each user’s directory within the dataset.
• A subset of these images (e.g., 25 per user) is selected randomly to ensure efficiency while maintaining sufficient data diversity.

Feature Extraction:
• Face Detection: Dlib’s frontal face detector is used to locate faces in the images.
• Landmark Detection: Dlib’s shape predictor 68 face landmarks detects 68
key facial landmarks to map the structural features of each face.
• Face Encoding: Each face’s unique features are encoded into a 128-dimensional vector using Dlib’s ResNet-based facial recognition model. These encodings represent the biometric signature of the face, which is later used for identification.

Storage:
• The computed face encodings and corresponding names are saved in memory.
• These are serialized into a file (trained model.pkl) for use during the recognition phase.
Output: A serialized model (trained model.pkl) containing facial encodings and associated names.

**3. RecognisingFaces.py**
Purpose: To recognize faces in real-time using the trained model and log attendance based on identified individuals.

Model Loading:
• The serialized model (trained model.pkl) containing precomputed face encodings is loaded into memory.

Real-Time Face Detection:
• A live video feed is captured from the webcam.
• Faces in the video are detected using Dlib’s frontal face detector.
• Detected faces are resized for faster processing, and their encodings are computed.

Recognition:
• Each computed encoding is compared against the known encodings from the model.
• The Euclidean distance between the encoding vectors is calculated.
• A threshold (e.g., 0.5) is used to determine a match.
• If a match is found, the name of the corresponding individual is retrieved.
• If no match is found, the face is labeled as ”Unknown.”

Attendance Logging:
• Recognized names, along with the current timestamp, are logged into a CSV file (attendance.csv).
• A record is added only if the individual’s attendance has not already been marked for the current day.
Live Feedback:
• The recognized name is displayed on the webcam feed alongside a bounding box around the detected face.
• Unknown faces are highlighted in red, while recognized faces are displayed in green.

Attendance View:
• A Flask web interface provides a real-time view of attendance logs by reading data
from the CSV file.
Output: Real-time facial recognition with attendance logs recorded in a CSV file

