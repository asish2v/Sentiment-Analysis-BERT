import joblib
from sklearn.ensemble import RandomForestClassifier  # Example model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline

# Example of creating a simple pipeline with a Random Forest Classifier
# If you used another model type, modify this accordingly

# This is just an example of model definition, you might have your trained model here
vectorizer = CountVectorizer()
model = make_pipeline(vectorizer, RandomForestClassifier())

# Train your model or load your pre-trained model here (if you already have one)

# For example, assuming you have some text data and corresponding labels:
# X_train = ['text1', 'text2', 'text3', ...]  # Your training text data
# y_train = [0, 1, 0, ...]  # Your labels (e.g., 0 for negative, 1 for positive)
# model.fit(X_train, y_train)

# Now, save the model
joblib.dump(model, 'sentiment_model.pkl')

print("Model saved successfully!")
