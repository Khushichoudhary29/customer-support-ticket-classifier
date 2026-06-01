# Customer Support Ticket Classification System

A machine learning-powered web application that automatically classifies customer support tickets into predefined categories using Natural Language Processing (NLP), Flask, and Scikit-Learn.

---

## Project Overview

Customer support teams often receive a large number of tickets every day. Manually categorizing these tickets can be time-consuming and inefficient.

This project automates the ticket classification process by predicting the most relevant category based on the ticket description. The system uses a machine learning model trained on customer support ticket data and provides predictions through an interactive web dashboard.

---

## Features

### Machine Learning Pipeline

* Text preprocessing and cleaning
* TF-IDF feature extraction
* Logistic Regression classifier
* Prediction confidence score
* Model persistence using Joblib

### Flask Backend

* REST API endpoints
* Real-time prediction service
* JSON response handling
* Error validation and handling

### Interactive Dashboard

* Modern Bootstrap user interface
* Ticket category prediction
* Confidence score visualization
* Prediction history table
* Dashboard statistics

### Database Integration

* SQLite database storage
* Persistent prediction history
* Automatic record management

### Data Export

* Export prediction history to CSV
* Download records for reporting and analysis

---

## Tech Stack

### Programming Language

* Python

### Machine Learning

* Scikit-Learn
* TF-IDF Vectorization
* Logistic Regression

### Backend Development

* Flask
* REST API

### Database

* SQLite

### Frontend

* HTML
* CSS
* Bootstrap 5
* JavaScript

### Version Control

* Git
* GitHub

---

## System Architecture

User Input

↓

Frontend Dashboard (HTML + Bootstrap)

↓

Flask REST API

↓

Machine Learning Model

↓

Prediction Result

↓

SQLite Database

↓

History Dashboard & CSV Export

---

## Project Structure

customer-support-ticket-classifier/

│

├── app.py

├── requirements.txt

├── README.md

│

├── model/

│ ├── train_model.py

│ ├── model.pkl

│ └── vectorizer.pkl

│

├── database/

│ ├── db.py

│ └── predictions.db

│

├── dataset/

│ └── tickets.csv

│

├── templates/

│ └── index.html

│

├── static/

│ └── style.css

│

└── screenshots/

---

## Installation

### Clone Repository

```bash
git clone https://github.com/your-username/customer-support-ticket-classifier.git

cd customer-support-ticket-classifier
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Application

### Initialize Database

```bash
python create_db.py
```

### Start Flask Server

```bash
python app.py
```

Application will be available at:

```text
http://127.0.0.1:5000
```

---

## API Endpoints

### Home Page

```http
GET /
```

Loads the dashboard.

---

### Predict Ticket

```http
POST /predict
```

Request:

```json
{
  "text": "I am unable to login to my account."
}
```

Response:

```json
{
  "prediction": "Login Issue",
  "confidence": 85.42
}
```

---

### Prediction History

```http
GET /history
```

Returns stored prediction records.

---

### Dashboard Statistics

```http
GET /stats
```

Returns:

```json
{
  "total_predictions": 100,
  "average_confidence": 82.5,
  "most_common_category": "Login Issue"
}
```

---

### Export History

```http
GET /export
```

Downloads prediction history as CSV.

---

## Machine Learning Workflow

### Data Collection

Customer support ticket dataset.

### Text Preprocessing

* Lowercase conversion
* Special character removal
* Whitespace normalization

### Feature Engineering

TF-IDF Vectorization converts text into numerical features.

### Model Training

Logistic Regression classifier is trained on processed ticket data.

### Prediction

The trained model predicts ticket categories and generates confidence scores.

---

## Current Limitations

The current implementation uses a synthetic customer support ticket dataset for demonstration and educational purposes.

Prediction accuracy depends heavily on dataset quality and diversity. Future versions will be trained on more realistic support ticket datasets to improve classification performance.

---

## Future Enhancements

* Deep Learning based classification
* BERT Transformer model integration
* User authentication and role management
* Ticket priority prediction
* Real-time analytics dashboard
* Cloud deployment using Render or AWS
* Docker containerization
* Admin panel for model monitoring

---

## Learning Outcomes

Through this project, I gained hands-on experience with:

* Natural Language Processing (NLP)
* Machine Learning Model Development
* Flask Backend Development
* REST API Design
* SQLite Database Integration
* Frontend Development
* Git and GitHub Workflow
* Full Stack ML Application Development

---

This project is developed for educational and portfolio purposes.
