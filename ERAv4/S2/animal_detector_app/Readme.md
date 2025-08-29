# Animal Detection Web App

A **Flask web application** that allows users to upload an image and detects whether the image contains a **cat, dog, or elephant** using a **pre-trained MobileNetV2 model** from Keras (ImageNet). The uploaded image is displayed on the webpage along with the detection results.

---

## Features

* Upload an image via browser.
* Detect cats, dogs, or elephants in the uploaded image.
* Display the uploaded image along with the detection results.
* Pre-trained MobileNetV2 model ensures high accuracy without training.
* Self-contained and easy to run locally.

---

## Folder Structure

```
animal_detector_app/
│
├── app.py                   # Main Flask application
├── uploads/                 # Uploaded images will be stored here
├── templates/
│   └── index.html           # HTML template for the web page
└── requirements.txt         # Python dependencies
```

---

## Setup Instructions

### 1. Clone the project or download the files

```bash
git clone <your-repo-url>
cd animal_detector_app
```

---

### 2. Create a Python 3.9 virtual environment

```bash
python3.9 -m venv venv
source venv/bin/activate  # Linux/macOS
# For Windows: venv\Scripts\activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt** should contain:

```
flask==2.3.4
tensorflow==2.15
keras==2.15
pillow==10.0.0
```

---

### 4. Run the Flask app

```bash
python app.py
```

* Server runs at: `http://127.0.0.1:5000`
* Open in your browser to access the web app.

---

### 5. Using the Web App

1. Open `http://127.0.0.1:5000` in your browser.
2. Upload a photo (JPG, PNG, etc.) using the upload form.
3. Wait a few seconds for the model to analyze the image.
4. The uploaded image will be displayed along with detection results (cat, dog, or elephant).

---

### 6. Notes

* Uses **MobileNetV2 pre-trained on ImageNet**, so it can detect a variety of classes.
* Filters predictions to only show **cat, dog, or elephant**.
* Uploaded images are stored in the `uploads/` folder.
* Can be easily extended to include more animal classes by modifying the filter in `app.py`.

---

### 7. Optional Enhancements

* Draw **bounding boxes** around detected objects.
* Support detection of **multiple animals** in one image.
* Add **history of uploads and predictions** in a CSV or database.
* Combine with **human gender detection** using DeepFace.
