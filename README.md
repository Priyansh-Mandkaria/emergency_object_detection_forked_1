# Emergency Vehicle Detection Web Application (Forked Version)

**Supervised by:** Dr. Dalia Nandi (Project Mentor), Professor at IIIT Kalyani

**Team Members:**  
- Priyansh Mandkaria (3rd year - CSE/22066)
- Nitin Pratap (3rd year - CSE/22061)
- Prince Kumar (3rd year - CSE/22064)
- Mrityunjay Kumar (3rd year - CSE/22057)

This repository is a fork of the original Emergency Vehicle Detection Web Application created by [Nitin Pratap](https://github.com/Nitinpratap22061/majorProjectcse). It represents collaborative work from our four-member team, contributing equally to the project.

---

## Overview

The Emergency Vehicle Detection Web Application is a real-time object detection system designed to identify emergency vehicles (e.g., ambulances, fire trucks, police cars) in uploaded videos or images. The system calculates the distance of the detected vehicle from the camera and sends an SMS alert to a specified phone number.

## Features

- Real-time Object Detection: Detects emergency and non-emergency vehicles using YOLO (You Only Look Once) model.  
- Distance Calculation: Uses bounding box dimensions to estimate the distance between the camera and the detected vehicle.  
- SMS Notifications: Sends a one-time SMS alert for each detected emergency vehicle using Twilio.  
- User-friendly Interface: Streamlit-based web application with easy upload and display functionality.

## Technologies Used

- Frontend: Streamlit  
- Backend: Python, OpenCV  
- Machine Learning: YOLOv5 model (ONNX format)  
- Messaging: Twilio API for SMS notifications

## Installation

Follow the steps below to set up and run the application:

### Prerequisites

- Python 3.8 or later installed  
- pip (Python package manager)

### Steps

1. Clone the repository:  
   `git clone https://github.com/Priyansh-Mandkaria/emergency_object_detection_forked_1`  
   `cd majorProjectcse`  
   `git checkout master`  # Switch to master branch if not already on it  

2. Install dependencies:  
   `pip install -r requirements.txt`

3. Add your Twilio credentials:  
   Update the Twilio configuration in `app.py`:  
   ```python
   TWILIO_ACCOUNT_SID = 'your_account_sid'
   TWILIO_AUTH_TOKEN = 'your_auth_token'
   TWILIO_PHONE_NUMBER = 'your_twilio_phone_number'
   RECIPIENT_PHONE_NUMBER = 'recipient_phone_number'
   
4. Download the YOLO model:
Ensure best.onnx is placed in the hell/weights directory.
Make sure the data.yaml file is correctly configured.

How to Run the Application
Run the Streamlit app locally: streamlit run app.py

Once the server starts, open the provided URL in your browser (usually http://localhost:8501).

Application Workflow
1. Upload Video or Image:
Upload a video (.mp4, .avi, .mov) or an image (.jpg, .jpeg, .png).

2. Object Detection:
The YOLO model detects emergency and non-emergency vehicles.
Bounding boxes and labels are displayed on the uploaded media.

3. Distance Calculation:
The system calculates the distance to each detected vehicle using bounding box dimensions and the Haversine formula.

4. SMS Alert:
If an emergency vehicle is detected, an SMS alert is sent once for each video or image.

Configuration
1. YOLO Model:
Model path: hell/weights/best.onnx
Config file: data.yaml

2. Twilio SMS:
Requires a valid Twilio account.
Update Twilio credentials in app.py.

3. Camera Calibration:
Adjust focal_length and real_world_width for accurate distance calculation.

**Screenshots :-**
1. Main Dashboard
![WhatsApp Image 2025-04-17 at 20 27 18_e60ccb55](https://github.com/user-attachments/assets/b314acf3-3c06-4015-8249-799847d38b05)


2. Video Detection : Demo Test
https://www.youtube.com/watch?v=cKI0leZuh8g
![image](https://github.com/user-attachments/assets/f302c7d7-e1be-4115-8b84-a8eeb59b7c74)


3. SMS Notification :-
   
![WhatsApp Image 2025-05-07 at 17 36 07_f4dd2f89](https://github.com/user-attachments/assets/a4425c63-8bf4-4773-bca7-b96029c245da)

4. Real Time Database :-
   
![WhatsApp Image 2025-05-07 at 17 36 07_7b50fdf6](https://github.com/user-attachments/assets/7b976355-967d-474c-bae2-fa681d340759)

   
**Future Improvements**
Multi-camera Support: Integrate feeds from multiple cameras for wider coverage.
Real-time Alerts: Extend the system for use with live video streams.
Enhanced Accuracy: Use advanced models like YOLOv8 for improved detection.

**Contact**
For any queries or contributions, please contact:

1 Nitin Pratap (Owner of repo)
- Email: pratapnitin87@gmail.com
- Email: cse22061@iiitkalyani.ac.in
- GitHub: https://github.com/Nitinpratap22061

2 Priyansh Mandkaria (Collaborator)
- Email: mandkariapriyansh@outlook.com
- Email: cse22066@iiitkalyani.ac.in
- Github: https://github.com/Priyansh-Mandkaria

3 Prince Kumar (Collaborator)
- Email: cse22064@iiitkalyani.ac.in
- Github: https://github.com/Userride

4 Mrityunjay Kumar (Collaborator)
- Email: cse22057@iiitkalyani.ac.in
- Github:

5 Dr. Dalia Nandi (Project Mentor)
- Email: dalianandi@iiitkalyani.ac.in
