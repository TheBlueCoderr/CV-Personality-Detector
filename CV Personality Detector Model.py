import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Updated sample dataset with all Big Five traits
data = {
    'resume_text': [
        "Led AI projects with strong leadership and high initiative",        # Openness
        "Worked well in teams, helped peers, showed empathy",               # Agreeableness
        "Consistently delivered results on time with precision",            # Conscientiousness
        "Created creative solutions and new strategies for ML problems",    # Openness
        "Handled stressful deadlines with emotional stability",             # Neuroticism
        "Comfortable in group presentations and public speaking roles",     # Extraversion
    ],
    'trait': [
        'Openness',
        'Agreeableness',
        'Conscientiousness',
        'Openness',
        'Neuroticism',
        'Extraversion'
    ]
}

df = pd.DataFrame(data)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['resume_text'])
y = df['trait']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
