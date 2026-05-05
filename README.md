# 🩺 AI Doctor Recommendation Chatbot

An AI-powered chatbot that recommends doctors based on user symptoms and location using NLP and Machine Learning.

---

## 🚀 Project Overview

This system takes user symptoms as input, processes them using NLP techniques, predicts the required medical specialization using a TensorFlow model, and recommends relevant doctors based on location.

---

## 🧠 Features

* Symptom-based doctor recommendation
* NLP preprocessing using NLTK
* Text vectorization using TF-IDF
* Deep Learning model using TensorFlow
* Location extraction using SpaCy
* Database integration using SQLite
* Chat-style web interface using Flask
* Smart validation to handle random/invalid inputs

---

## ⚙️ Technologies Used

* Python
* Flask
* TensorFlow
* Scikit-learn
* Pandas
* NLTK
* SpaCy
* SQLite
* HTML/CSS

---

## 🧩 How It Works

1. User enters symptoms
2. Input is cleaned using NLP (tokenization, stopwords removal)
3. TF-IDF converts text into numerical form
4. TensorFlow model predicts specialization
5. SpaCy extracts city from input
6. Database is queried using specialization + city
7. Recommended doctors are displayed

---

## 📊 Dataset

### Symptoms Dataset

Maps symptoms to medical specialization

### Doctors Dataset

Contains:

* Name
* Specialization
* Hospital
* City
* Availability

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/mubashir-azeem/AI-Doctor-Recommendation-Chatbot-NLP-TensorFlow-Flask.git
```

### 2. Navigate to the project folder

```bash
cd AI-Doctor-Recommendation-Chatbot-NLP-TensorFlow-Flask
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup the database

```bash
python database.py
```

### 5. Run the application

```bash
python app.py
```

### 6. Open in browser

http://127.0.0.1:5000/

---

## 🎯 Example Inputs

* chest pain in karachi
* migraine islamabad
* skin rash lahore

---

## 💡 Key Learning

This project demonstrates how NLP, Machine Learning, and Web Development can be integrated to build real-world AI applications.

---

## 🔗 Author

Developed by Mubashir Azeem
AI/ML Internship (Remote) at BlackByte
