# CV Personality Detector 🧠📄

A Natural Language Processing (NLP) project that predicts Big Five personality traits (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism) based on resume content. This project uses **TF-IDF vectorization** and **Logistic Regression** to classify text samples from resumes into personality categories.

---

## 🧩 Problem Statement

Hiring decisions often depend on personality traits in addition to technical skills. This project explores how textual patterns in resumes can be used to infer psychological traits using traditional NLP techniques.

---

## 🚀 Features

- TF-IDF vectorization of resume text
- Multiclass classification using Logistic Regression
- Basic evaluation with precision, recall, and F1-score
- Ready to be extended with Word2Vec/BERT or real datasets

---

## 🧠 Tech Stack

- Python
- scikit-learn
- pandas
- TF-IDF Vectorization
- Logistic Regression

---

## 📁 Dataset

For demonstration, a small simulated dataset is used. You can replace it with a real labeled dataset with columns like:
```csv
resume_text, trait
"Handled team leadership...", Openness
"Worked under pressure...", Neuroticism 
```
## ⚙️ How to Run
Clone the repository
git clone https://github.com/yourusername/cv-personality-detector.git
cd cv-personality-detector

Install required libraries
pip install -r requirements.txt

Run the project
python main.py


## 📊 Example Output
python-repl
Copy
Edit
Classification Report:

              precision    recall  f1-score   support

Openness         1.00      1.00      1.00       1
...


## 🛠️ Future Improvements
Use Word2Vec or BERT embeddings

Train on a larger real-world dataset

Deploy with a Streamlit or Flask front-end
