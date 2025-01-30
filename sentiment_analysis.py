import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the dataset
data = pd.read_csv('amazon_reviews.csv')

# Step 2: Check the column names to identify the review and sentiment columns
print(data.columns)  # This prints all the column names to help identify the ones you need

# Step 3: Preprocess the review text (convert to lowercase)
data['review'] = data['reviews.text'].str.lower()  # Convert review text to lowercase

# Step 4: Convert ratings to binary labels (1 = positive, 0 = negative)
# Assuming ratings >= 4 are positive and below 4 are negative
data['label'] = data['reviews.rating'].apply(lambda x: 1 if x >= 4 else 0)

# Step 5: Feature Extraction - Convert text data to TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)  # Limit to top 5000 features
X = vectorizer.fit_transform(data['review'])  # Convert the reviews to a matrix of TF-IDF features
y = data['label']  # The sentiment labels

# Step 6: Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 8: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 9: Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("Classification Report:\n", classification_report(y_test, y_pred))
