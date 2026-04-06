![App Screenshot](screenshot.png)

# 🩻 Chest X-ray Pneumonia Detection

This Flask web app allows users to upload chest X-ray images and detect Pneumonia using a pre-trained deep learning model in TensorFlow Lite format.

##  Features

- Upload a patient's chest X-ray image
- Select a model from a list of available `.tflite` models
- Perform inference using the selected model

## 📂 Folder Structure


## 🚀 How to Run Locally

### 1. Clone this Repository
```bash
git clone https://github.com/elaanba/xray-diagnosis-app-1.git
cd xray-diagnosis-app

###### OR ############

Set Up a Virtual Environment

python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

Install Dependencies


pip install -r requirements.txt


Run the Flask App

python app.py





This app uses classification models that detect Pneumonia
