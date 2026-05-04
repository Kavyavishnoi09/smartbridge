# Rice Type Classification System

This project is a Deep Learning-based web application that classifies rice grain images into five categories using MobileNet and Flask.
<img width="1907" height="732" alt="Screenshot 2026-03-31 003407" src="https://github.com/user-attachments/assets/402c77ef-7bca-47ad-91dd-549158f22a77" />
<img width="1603" height="896" alt="Screenshot 2026-03-31 010535" src="https://github.com/user-attachments/assets/7dc04d0f-7c66-45f8-bf3b-8ce2bc8eab3b" />
  <img width="1202" height="912" alt="image" src="https://github.com/user-attachments/assets/e793ad7b-1e8d-4495-8da7-a807aa97f541" />


## Project Overview

The objective of this project is to identify the type of rice from an uploaded image.  
The system predicts one of the following classes:

- Arborio
- Basmati
- Ipsala
- Jasmine
- Karacadag

The model was trained using TensorFlow/Keras and integrated into a Flask web application for real-time prediction.

## Technologies Used

- Python
- TensorFlow / Keras
- MobileNet
- Flask
- HTML / CSS
- NumPy
- Werkzeug

## Model Details

- Model Architecture: MobileNet
- Input Image Size: 160 x 160
- Number of Classes: 5
- Validation Accuracy: ~98%

## Project Structure

```text
GrainPalette/
│
├── app.py
├── predict.py
├── README.md
├── requirements.txt
│
├── training/
│   └── rice.h5
│
├── templates/
│   ├── index.html
│   ├── details.html
│   └── results.html
│
├── static/
│   ├── css/
│   └── uploads/
│
└── data/






