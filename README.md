# Fatigue Detection System

The **Fatigue Detection System** is a video-based drowsiness and alertness monitoring tool. It detects visual signs of fatigue by analyzing a subject’s eye and mouth state across time. The system uses a subsampled-frame convolutional neural network (CNN) pipeline and computes two established fatigue indicators: **PERCLOS** (Percentage of Eye Closure) and **POM** (Percentage of Mouth Opening).

If PERCLOS or POM exceeds a defined threshold, the system classifies the subject as fatigued.

---

## Table of Contents
- [Overview](#overview)
- [How Fatigue Is Detected](#how-fatigue-is-detected)
- [Technologies Used](#technologies-used)
- [System Workflow](#system-workflow)
- [Metrics](#metrics)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Docker Usage](#docker-usage)
- [Notes](#notes)

---

## Overview

This system takes a video input (for example, a student during an online class, or a driver in a cabin camera) and analyzes visible facial states over time. It focuses on:

- **Eyes**: open vs. closed  
- **Mouth**: open vs. closed (e.g. yawning behavior)

The pipeline:

1. Extracts frames from the input video (subsampled, not every frame).
2. Detects faces using **MTCNN (Multi-task Cascaded Convolutional Networks)**.
3. Crops and preprocesses the regions of interest (eyes and mouth).
4. Classifies each region using a trained CNN.
5. Aggregates results over time to compute PERCLOS and POM.
6. Decides if the subject is fatigued.

---

## How Fatigue Is Detected

We use two core behavioral metrics:

1. **PERCLOS (Percentage of Eye Closure)**  
   - Measures how often the eyes are closed over the observed interval.  
   - Higher PERCLOS → higher likelihood of drowsiness.

2. **POM (Percentage of Mouth Opening)**  
   - Measures how often the mouth is open (e.g. yawning).  
   - Higher POM → higher likelihood of fatigue.

**Decision rule (example thresholding):**  
- If **PERCLOS > 50%** OR **POM > 50%**, classify the subject as **Fatigued**.  
- Otherwise, classify as **Alert**.

You can tune these thresholds depending on the use case.

---

## Technologies Used

- **Python**
- **TensorFlow / Keras** — CNN model for eye/mouth state classification  
- **OpenCV** — video frame extraction, subsampling, preprocessing  
- **MTCNN** — face / facial region (eyes, mouth) detection  
- **Flask** — web interface for uploading and evaluating videos

---

## System Workflow

### 1. Frame Extraction / Subsampling
Instead of processing every frame, the system samples frames at intervals. This reduces compute cost while still capturing temporal behavior. Subsampling can also be filtered by energy/activity to ignore low-information frames.

### 2. Face and Landmark Detection
For each kept frame:
- Detect face using **MTCNN**
- Locate eyes and mouth regions
- Crop and resize those regions for inference

### 3. CNN-Based State Classification
A trained CNN predicts:
- Eyes: **Open** or **Closed**
- Mouth: **Open** or **Closed**

These predictions are stored for each analyzed frame.

### 4. Metric Computation
From all analyzed frames:
- **PERCLOS** = % of frames where eyes are classified closed  
- **POM** = % of frames where mouth is classified open

### 5. Fatigue Decision
If either metric crosses the configured threshold (e.g. 50%), the system flags fatigue.

---

## Metrics

**PERCLOS (Percentage of Eye Closure)**  
- Fraction of time (or frames) where the eyes are closed.

**POM (Percentage of Mouth Opening)**  
- Fraction of time (or frames) where the mouth is open.

These two values are reported at the end of processing, and they drive the final fatigue/no-fatigue decision.

---

## Directory Structure

```plaintext
fatigue_detection_app/
│
├── models/                     # Pre-trained model(s)
│   └── fatigue_model.h5        # CNN model for eye/mouth state classification
│
├── static/                     # Static assets (CSS, JS, etc.) for Flask
│
├── templates/                  # HTML templates for Flask views
│   ├── index.html              # Upload interface for the video
│   └── result.html             # Results page (PERCLOS, POM, fatigue status)
│
├── uploads/                    # Uploaded video files are stored here
│
├── utils/                      # Processing utilities
│   ├── model_utils.py          # Model loading, prediction helpers
│   └── dataset.py              # Dataset / preprocessing helpers
│
├── app.py                      # Main Flask application entry point
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation (this file)
└── .gitignore
```

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/fatigue_detection_app.git
cd fatigue_detection_app
```

### 2. Create and activate a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate   # macOS / Linux
# OR
venv\Scripts\activate    # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

1. **Start the Flask app**
   ```bash
   python app.py
   ```

2. **Open the web UI**  
   Go to:  
   `http://localhost:5000`

3. **Upload a video**  
   On the home page, upload a video where the face is clearly visible.

4. **Processing**  
   The system will:
   - Subsample frames from the video  
   - Detect the face / eyes / mouth  
   - Run the CNN model on each region  
   - Compute PERCLOS and POM

5. **View Results**  
   After processing, you'll see:
   - PERCLOS value  
   - POM value  
   - Final classification: **Fatigued** or **No Fatigue Detected**

---

## Docker Usage

You can also run the app in a container instead of installing everything locally.

### 1. Build the Docker image
```bash
docker build -t fatigue-detection-app .
```

### 2. Run the container
```bash
docker run -p 5000:5000 fatigue-detection-app
```

Now open `http://localhost:5000` in your browser.

### 3. Using Docker Compose
If you have a `docker-compose.yml`:
```bash
docker-compose up
```

This will also expose the app on `http://localhost:5000`.

---

## Notes

- The thresholds for PERCLOS and POM (50%) are defaults; in production with real users, you'd calibrate based on your domain (e.g. driving, exam monitoring, online learning).
- For best results, videos should have good lighting and a clear, mostly frontal face.
- The current workflow assumes a single subject in frame.

---
