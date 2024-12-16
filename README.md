Face Recognition System
Overview
This project is a Face Recognition System built using Python, OpenCV, and MySQL. It allows users to register their faces, train a classifier, and recognize faces in real-time using a webcam. The system is designed to be user-friendly and provides a graphical interface for easy interaction.

Features
User Registration: Users can input their name, age, and address to register their face.
Face Detection: The system uses Haar Cascade Classifier for detecting faces in real-time.
Face Recognition: Trained models can recognize registered users and display their names.
Database Integration: User data is stored in a MySQL database for persistence.
Image Capture: The system captures images of the user's face for training the model.
Requirements
Python 3.11
OpenCV
MySQL Connector
Tkinter
PIL (Pillow)
NumPy
Installation
Clone the repository:

bash

Verify

Open In Editor
Run
Copy code
git clone <repository-url>
cd face-recognition
Install the required packages:

bash

Verify

Open In Editor
Run
Copy code
pip install opencv-python mysql-connector-python pillow numpy
Set up the MySQL database:

Create a database named pro_student.
Run the SQL commands in connectdb.py to create the necessary tables.
Download the Haar Cascade XML file:

Ensure that haarcascade_frontalface_default.xml is in the project directory.
Usage
Run the application:

bash

Verify

Open In Editor
Run
Copy code
python nasfr_System.py
The GUI will open. You can:

Train the Classifier: Click on the "Training" button to train the model with the captured images.
Generate Dataset: Click on "Generate Dataset" to capture images of your face for training.
Detect Face: Click on "Detect Face" to start the real-time face recognition.
Follow the prompts in the GUI to complete the registration and recognition process.

Contributing
Contributions are welcome! If you have suggestions for improvements or new features, feel free to open an issue or submit a pull request.

Acknowledgments
OpenCV for the face detection and recognition algorithms.
MySQL for data storage.
Tkinter for the graphical user interface.
Contact
For any inquiries, please contact Aditya Kadam at Kadamaditya2020@gmail.com
