# 🫁 Pneumonia Detection System

[![GitHub](https://img.shields.io/badge/GitHub-ASriprahas-181717?style=flat&logo=github)](https://github.com/ASriprahas)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Sri%20Prahas%20Annambhotla-0A66C2?style=flat&logo=linkedin)](https://www.linkedin.com/in/sri-prahas-0396b1330/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=flat&logo=pytorch)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-ML%20Service-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Spring Boot](https://img.shields.io/badge/Spring%20Boot-3.x-6DB33F?style=flat&logo=springboot)](https://spring.io/projects/spring-boot)
[![React](https://img.shields.io/badge/React-Vite%20%2B%20Tailwind-61DAFB?style=flat&logo=react)](https://react.dev/)

A full-stack, AI-powered clinical decision-support web application that detects pneumonia from chest X-ray radiographs using deep learning and provides visual model explainability via **Grad-CAM** (Gradient-weighted Class Activation Mapping).

---

## 📋 Table of Contents
- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Deep Learning Models & Explainability](#-deep-learning-models--explainability)
- [Repository Structure](#-repository-structure)
- [Prerequisites](#-prerequisites)
- [Installation & Setup](#-installation--setup)
  - [1. Database Setup (MySQL)](#1-database-setup-mysql)
  - [2. ML Inference Service (FastAPI)](#2-ml-inference-service-fastapi)
  - [3. Backend API (Spring Boot)](#3-backend-api-spring-boot)
  - [4. Frontend Client (React + Vite)](#4-frontend-client-react--vite)
- [API Endpoints](#-api-endpoints)
- [Disclaimer](#-disclaimer)
- [Author](#-author)

---

## 🔍 Overview

Diagnosing pneumonia from chest X-rays requires careful radiological interpretation and can be time-consuming in high-volume clinical settings. This system bridges deep learning inference with a responsive web interface, allowing users to upload chest X-ray images, receive automated classification predictions (**Normal** vs. **Pneumonia**) along with confidence scores, inspect localized lung opacity heatmaps via **Grad-CAM**, and persist diagnostic records in a relational database.

### Key Features
- **Multi-Model Deep Learning Inference:** Supports predictions across transfer-learning backbones (**VGG16**, **ResNet50**) and a custom 4-block CNN (**PneumoniaNet**).
- **Grad-CAM Visual Explainability:** Generates overlaid heatmap visualizations highlighting the exact thoracic regions influencing the model's prediction.
- **Decoupled Microservice Architecture:** Separates the React frontend, Java Spring Boot business/persistence layer, and Python FastAPI PyTorch inference engine.
- **Diagnostic History & Persistence:** Stores scan metadata, model selections, confidence metrics, and results in MySQL (`pneumonia_db`).
- ---

## 🏗 System Architecture

1. **Client Layer (`Frontend/`):** Users upload a chest X-ray image (JPEG/PNG) and choose an inference model.
2. **Application Layer (`Backend/`):** Spring Boot validates the upload, communicates with the ML microservice, and persists diagnostic logs in MySQL.
3. **Inference Layer (`ML-service/`):** FastAPI preprocesses the radiograph, runs PyTorch inference, computes Grad-CAM activation maps, and returns the prediction, confidence score, and heatmap overlay.

---

## 🛠 Tech Stack

| Layer | Technologies |
| :--- | :--- |
| **Frontend** | React, Vite, Tailwind CSS, Axios |
| **Backend** | Java 17+, Spring Boot 3, Spring Data JPA, Maven |
| **ML Microservice** | Python 3.10+, FastAPI, Uvicorn, PyTorch, Torchvision, OpenCV, NumPy, Pillow |
| **Database** | MySQL 8.0 (`pneumonia_db`) |

---

## 🧠 Deep Learning Models & Explainability

The ML service evaluates chest radiographs across three architectures trained on labeled chest X-ray datasets (`NORMAL` vs. `PNEUMONIA`):

1. **VGG16 (Transfer Learning):** Pre-trained convolutional feature extractor fine-tuned with a custom binary classification head.
2. **ResNet50 (Transfer Learning):** Deep residual architecture utilizing skip connections to capture subtle pulmonary opacities and consolidation patterns.
3. **PneumoniaNet (Custom CNN):** A lightweight, domain-specific 4-block Convolutional Neural Network designed for fast inference and low parameter overhead.
4. **Grad-CAM Explainability:** Extracts gradients flowing into the final convolutional layer to generate a class-discriminative localization heatmap, enhancing transparency during clinical review.

---

## 📁 Repository Structure

```text
pneumonia-detection-system/
├── Backend/                   # Java Spring Boot 3 REST API & MySQL persistence
├── Frontend/                  # React + Vite + Tailwind CSS client application
├── ML-service/                # Python FastAPI + PyTorch inference & Grad-CAM service
├── .gitignore                 # Excludes model weights, venv, uploads, and build artifacts
└── README.md                  # Project documentation
```

---

## ✅ Prerequisites

Ensure the following tools are installed before running the project locally:
- **Node.js** (v18+) & **npm**
- **Java Development Kit (JDK)** (v17+) & **Maven**
- **Python** (v3.10+) & **pip**
- **MySQL Server** (v8.0+)

---

## 🚀 Installation & Setup

### Clone the Repository

```bash
git clone https://github.com/ASriprahas/pneumonia-detection-system.git
cd pneumonia-detection-system
```
### 1. Database Setup (MySQL)

Create the MySQL database:

```sql
CREATE DATABASE pneumonia_db;```

Configure your database credentials in `Backend/src/main/resources/application.properties`:

```properties
server.port=8080
spring.datasource.url=jdbc:mysql://localhost:3306/pneumonia_db?useSSL=false&serverTimezone=UTC
spring.datasource.username=YOUR_MYSQL_USERNAME
spring.datasource.password=YOUR_MYSQL_PASSWORD
spring.jpa.hibernate.ddl-auto=update
```

### 2. ML Inference Service (FastAPI)

```bash
cd ML-service
python -m venv venv

# Activate virtual environment (Windows)
venv\Scripts\activate
# Activate virtual environment (macOS/Linux)
source venv/bin/activate

pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

> **Note:** Trained PyTorch model weight files (`.pth`/`.pt`) are excluded from Git via `.gitignore`. Place your trained weight files in the expected model directory inside `ML-service/` before running inference.

### 3. Backend API (Spring Boot)

Open a new terminal and start the Spring Boot server on port `8080`:

```bash
cd Backend
./mvnw spring-boot:run
```

*(On Windows CMD, run `mvnw.cmd spring-boot:run`)*

### 4. Frontend Client (React + Vite)

Open a new terminal and start the React development server:

```bash
cd Frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser to use the application.

---

## 🔌 API Endpoints

### Spring Boot Backend (`http://localhost:8080`)

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/predictions/analyze` | Uploads a chest X-ray, invokes ML inference, saves the record, and returns the prediction + Grad-CAM heatmap |
| `GET` | `/api/predictions/history` | Fetches past diagnostic records stored in `pneumonia_db` |
| `GET` | `/api/predictions/{id}` | Retrieves details for a specific diagnostic scan |

### FastAPI ML Service (`http://localhost:8000`)

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Verifies active service status, loaded PyTorch models, and compute device (`CPU`/`CUDA`) |
| `POST` | `/predict` | Accepts an image and model choice (`vgg16`, `resnet50`, `pneumonianet`), returning predicted class, confidence, and Grad-CAM overlay |

---

## ⚠️ Disclaimer

This project is developed strictly for **academic, research, and educational purposes**. It is not a certified medical device and must not be used as a substitute for professional medical diagnosis or clinical decision-making.

---

## 👨‍💻 Author

**Sri Prahas Annambhotla**  
B.Tech – Computer Science and Business Systems  
VIT-AP University  

- **GitHub:** [https://github.com/ASriprahas](https://github.com/ASriprahas)
- **LinkedIn:** [https://www.linkedin.com/in/sri-prahas-0396b1330/](https://www.linkedin.com/in/sri-prahas-0396b1330/)

