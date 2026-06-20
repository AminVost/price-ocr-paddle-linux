# PaddleOCR Price Extraction Pipeline (Linux Optimized)

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PaddlePaddle](https://img.shields.io/badge/PaddlePaddle-%230075FF.svg?style=for-the-badge&logo=paddlepaddle&logoColor=white)
![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)

An automated, high-performance Optical Character Recognition (OCR) pipeline engineered to extract tabular data and pricing information from complex document images. This specific build is strictly optimized for headless Linux server environments.

## 🎯 Business Problem & Solution
Manually transcribing pricing data from supplier catalogs, hardware tool price lists, or dense invoices is highly error-prone and time-consuming. 

This solution leverages the robust architecture of **PaddleOCR** to accurately detect and read dense tabular data. It converts raw images into structured, machine-readable formats (JSON/CSV) for seamless integration into enterprise ERP systems or automated pricing databases.

## 🚀 Key Features
* **High-Accuracy Tabular Extraction:** Fine-tuned to detect complex price list structures, distinct columns, and numeric data accurately.
* **Linux Deployment Ready:** Pre-configured to run efficiently on Linux servers without heavy GUI dependencies. Solves common library conflicts (e.g., `libGL.so.1` missing errors) often encountered during server deployments.
* **Automated Data Structuring:** Includes post-processing logic that aligns detected bounding boxes into coherent logical rows and columns.
* **Strictly English Codebase:** All code, inline documentation, and operational logs are maintained exclusively in English to meet strict international enterprise CI/CD standards.

## 🏛 Technical Architecture
* **Core OCR Engine:** PaddleOCR (built on the PaddlePaddle deep learning framework).
* **Image Preprocessing:** OpenCV (`cv2`) and NumPy for image binarization, deskewing, and noise reduction prior to model inference.
* **Data Parsing:** Pandas for structuring the raw OCR output into clean datasets.

## 💻 Environment Setup (Linux)
*(Internal deployment instructions. Requires specific Linux packages such as `libgl1-mesa-glx` and `libglib2.0-0` for OpenCV headless operation).*
