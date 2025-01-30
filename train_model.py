import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Load dataset
data = pd.read_csv('amazon_reviews.csv')

# Preprocess the data
data['review'] = data['reviews.text'].str.lower()  # Convert reviews to lowercase
data['label'] = data['reviews.rating'].apply(lambda x: 1 if x >= 4 else 0)  # Assign label: 1=positive, 0=negative

# Features and Labels
X = data['review']
y = data['label']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with TF-IDF and Random Forest classifier
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),  # Convert text to numeric vectors
    ('clf', RandomForestClassifier())  # Use a Random Forest model
])

# Train the model
pipeline.fit(X_train, y_train)

# Save the model to a file
joblib.dump(pipeline, 'sentiment_model.pkl')

# Check accuracy
accuracy = pipeline.score(X_test, y_test)
print(f"Model Accuracy: {accuracy}")
