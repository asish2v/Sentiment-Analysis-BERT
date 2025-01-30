import streamlit as st
from transformers import pipeline

# Load the pre-trained sentiment-analysis model from Hugging Face
nlp = pipeline("sentiment-analysis")

# Title of the app
st.title("Sentiment Analysis with BERT")

# Description
st.write("""
    This app uses a pre-trained BERT model to predict the sentiment of your review.
    Enter a product review below, and it will classify it as positive or negative.
""")

# Text input for user to enter a review
review_text = st.text_area("Enter Review Text")

# Button for prediction
if st.button("Predict Sentiment"):
    if review_text:
        # Predict sentiment using the BERT model
        result = nlp(review_text)

        # Display the result
        st.write(f"Prediction Raw Output: {result}")
        
        # Display sentiment as positive or negative
        sentiment = result[0]['label']
        if sentiment == 'POSITIVE':
            st.success("This is a Positive Review!")
        else:
            st.error("This is a Negative Review!")
    else:
        st.warning("Please enter a review text to analyze.")
