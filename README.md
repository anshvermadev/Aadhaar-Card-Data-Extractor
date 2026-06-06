# Aadhaar Card Data Extractor

[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8-orange.svg)](https://github.com/ultralytics/ultralytics)
[![Tesseract OCR](https://img.shields.io/badge/OCR-Tesseract-red.svg)](https://github.com/tesseract-ocr/tesseract)

An intelligent, full-stack hybrid AI application that automates the extraction of key demographic fields from Aadhaar cards (India's unique identity card). It utilizes a custom-trained **YOLOv8** object detection model to locate text regions, applies advanced **OpenCV** preprocessing filters for text clarity, and runs **Tesseract OCR** to parse the content with high-accuracy fallbacks.

---

## 📖 Table of Contents

- [What the Project Does](#-what-the-project-does)
- [Tech Stack](#-tech-stack)
- [Why the Project is Useful](#-why-the-project-is-useful)
- [How Users Can Get Started](#-how-users-can-get-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Model Training Details](#model-training-details)
  - [Running the Application](#running-the-application)
  - [API Usage Example](#api-usage-example)
- [Where Users Can Get Help](#-where-users-can-get-help)

---

## 🔍 What the Project Does

The **Aadhaar Card Data Extractor** is a hybrid computer vision and OCR pipeline designed to process images of Aadhaar cards and return structured JSON data. It extracts the following fields:
*   **Full Name** (`NAME`)
*   **Date of Birth** (`DATE_OF_BIRTH`)
*   **Gender** (`GENDER`)
*   **Aadhaar Number** (`AADHAR_NUMBER`) - Cleaned and formatted as `XXXX XXXX XXXX`

### System Architecture
The application is split into a modular backend and an interactive web interface:
1.  **FastAPI Backend ([main.py](main.py))**: A high-performance REST API that serves model predictions, performs image preprocessing, runs regional Tesseract OCR, and executes post-processing cleanup and verification regex.
2.  **Modern Web Frontend ([index.html](index.html))**: A clean, single-page UI built with HTML5, CSS3, and Vanilla JS. It features a drag-and-drop upload zone, interactive loading progress indicators, and a clean interface to copy individual fields or all results to the clipboard.

---

## 🛠️ Tech Stack

This project is built using a modern, lightweight, and high-performance stack for deep learning, computer vision, and web APIs:

### Backend & API
*   **[FastAPI](https://fastapi.tiangolo.com/)**: A modern, fast (high-performance) web framework for building APIs with Python 3.8+ based on standard Python type hints.
*   **[Uvicorn](https://www.uvicorn.org/)**: A lightning-fast ASGI server implementation for Python.

### Machine Learning & Computer Vision
*   **[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)**: Used for high-precision region of interest (ROI) detection to find demographic field bounding boxes on Aadhaar cards.
*   **[PyTorch](https://pytorch.org/)**: The underlying deep learning backend powering the YOLOv8 model inference.

### OCR & Image Processing
*   **[Tesseract OCR](https://github.com/tesseract-ocr/tesseract)** / **[PyTesseract](https://github.com/madmaze/pytesseract)**: Used to perform Optical Character Recognition on localized regions of interest.
*   **[OpenCV-Python-Headless](https://opencv.org/)**: Handles high-performance image manipulation, resizing, denoising, sharpening, and binarization filters.
*   **[Pillow (PIL)](https://python-pillow.org/)** & **[NumPy](https://numpy.org/)**: Used for image array representation and memory manipulation.

### Frontend
*   **HTML5 & CSS3 (Vanilla)**: Features custom variables (custom dark/light surface system), responsive glassmorphic cards, flat buttons, and keyframe animations.
*   **Vanilla JavaScript (ES6+)**: Employs reactive DOM updates, async file readers, drag-and-drop event listeners, and Fetch API integrations.

---

## 💡 Why the Project is Useful

Demographic data extraction from identity documents is a crucial step in digital onboarding, KYC (Know Your Customer) workflows, and user registration. This project provides several key advantages:

*   **Hybrid Deep Learning + OCR Pipeline**: Rather than running OCR on the entire card (which introduces noise from decorative borders, logos, and multi-lingual translations), it uses **YOLOv8** to target only the regions containing the relevant data.
*   **High-Resolution Cropping**: Bounding boxes predicted on the 640x640 YOLO grid are dynamically mapped and cropped from the *original high-resolution image*, ensuring that Tesseract receives sharp, uncompressed text characters.
*   **Advanced Preprocessing Suite**: The backend uses OpenCV to enhance text legibility in cropped regions:
    *   *Sharpening Filter*: Highlights character edges to resolve blurry text.
    *   *Fast Non-Local Means Denoising*: Removes salt-and-pepper camera noise without blurring text borders.
    *   *Otsu's Binarization*: Automatically binarizes text regions to black-and-white.
    *   *Bespoke White-listing*: Restricts Tesseract characters based on the target class (e.g., only numbers/spaces for Aadhaar numbers, numbers/slashes for DOB).
*   **Robust Fallback Mechanisms**: If the YOLOv8 model fails to detect a field with high confidence (e.g., due to poor lighting or rotation), the API automatically falls back to:
    1.  **Full-Card OCR Text Scan**: Broad parsing across the entire card using regex pattern matching.
    2.  **Targeted Bottom-Half OCR**: Specifically sweeps the lower 40% center region where the 12-digit Aadhaar number is standardly placed.
*   **Production-Ready UI**: A highly responsive, visual interface that handles errors gracefully, features stage-by-stage pipeline feedback (Preprocessing ➡️ YOLOv8 ➡️ OCR ➡️ Extraction), and implements a one-click copy helper.

---

## 🚀 How Users Can Get Started

### Prerequisites

1.  **Python 3.9+** (Tested on Python 3.12)
2.  **Tesseract OCR Engine**:
    *   **Windows**: Download and install from [UB Mannheim's Tesseract Installer](https://github.com/UB-Mannheim/tesseract/wiki).
    *   **Linux (Ubuntu/Debian)**: `sudo apt-get install tesseract-ocr`
    *   **macOS**: `brew install tesseract`
    *   *Note for Windows users:* The backend has a default Tesseract path configured at `C:\Program Files\Tesseract-OCR\tesseract.exe`. If you install it elsewhere, modify the `pytesseract.tesseract_cmd` setting on line 12 of `main.py`.

### Installation

1.  **Install Dependencies**:
    Install all required Python libraries using pip:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Verify Model Placement**:
    Ensure that the custom trained weights file [yolov8_trained_model.pt](yolov8_trained_model.pt) is located in the root of the project directory.

### Model Training Details

The YOLOv8 small model (`yolov8s.pt`) was trained on a dataset of Aadhaar cards to detect 5 classes (representing text areas for Name, Date of Birth, Gender, and Aadhaar Number). 
*   **Dataset Source**: [Kaggle Aadhaar Dataset](https://www.kaggle.com/datasets/nagendra048/aadhar-dataset)
*   **Training Script/Process**: Refer to the Jupyter Notebook [notebookebf6259eb7.ipynb](notebookebf6259eb7.ipynb) which documents the Kaggle GPU training run using PyTorch, AdamW optimizer, and custom augmentations over 100 epochs.

### Running the Application

1.  **Start the FastAPI Backend**:
    Run the Uvicorn server:
    ```bash
    python main.py
    ```
    Alternatively, launch it with reload enabled for development:
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    ```
    The API docs will be available at `http://localhost:8000/docs`.

2.  **Open the Frontend**:
    Since the frontend is a standalone, client-side application, you can simply double-click [index.html](index.html) to open it in any web browser, or serve it using an extension like VS Code Live Server.

### API Usage Example

You can integrate the extractor API into other scripts or apps. Send a `POST` request to the `/extract` endpoint with the image file:

#### Request (Python)
```python
import requests

url = "http://localhost:8000/extract"
files = {"file": open("path_to_aadhar_image.jpg", "rb")}

response = requests.post(url, files=files)
data = response.json()

if data.get("success"):
    print("Extracted Details:")
    print(f"Name: {data['extracted']['NAME']}")
    print(f"DOB:  {data['extracted']['DATE_OF_BIRTH']}")
    print(f"Gender: {data['extracted']['GENDER']}")
    print(f"Aadhaar Number: {data['extracted']['AADHAR_NUMBER']}")
```

#### Response JSON
```json
{
  "success": true,
  "extracted": {
    "NAME": "ARAVIND SHARMA",
    "DATE_OF_BIRTH": "15/08/1990",
    "GENDER": "Male",
    "AADHAR_NUMBER": "1234 5678 9012"
  },
  "all_detections": [
    {
      "label": "NAME",
      "text": "ARAVIND SHARMA",
      "raw_text": "ARAVIND SHARMA\n",
      "confidence": 0.892,
      "bbox": [220, 140, 480, 185]
    }
  ],
  "model_classes": ["AADHAR_NUMBER", "DATE_OF_BIRTH", "GENDER", "NAME", "NAME"]
}
```

---

## 🙋 Where Users Can Get Help

If you run into issues or have questions:
*   **Issues**: Open a ticket on our GitHub Issues page.
*   **Documentation**:
    *   For FastAPI configurations, refer to the [FastAPI Documentation](https://fastapi.tiangolo.com/).
    *   For YOLOv8 fine-tuning, visit [Ultralytics Docs](https://docs.ultralytics.com/).
    *   For Tesseract OCR optimization, visit [Tesseract Docs](https://tesseract-ocr.github.io/tessdoc/).
