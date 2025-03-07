# 📌 Import necessary libraries
import streamlit as st  # For creating the web app
import pandas as pd  # For handling the dataset
import joblib  # For loading the trained model
from sklearn.feature_extraction.text import TfidfVectorizer  # For text processing
from sklearn.linear_model import LogisticRegression  # Machine Learning model
from sklearn.model_selection import train_test_split  # Splitting data
from sklearn.metrics import accuracy_score  # Model evaluation

# 🔹 Function to load dataset from GitHub
@st.cache_resource  # Caches the data to avoid reloading it every time
def load_data():
    # URL to the dataset stored in your GitHub repository
    url = "https://raw.githubusercontent.com/sriharshaganjam/sentiment-analysis/main/Reviews.csv"

    # Load the dataset
    df = pd.read_csv(url, encoding='latin-1')

    # Select relevant columns: 'Text' for reviews and 'Score' for ratings
    df = df[['Text', 'Score']]

    # Drop missing values
    df = df.dropna()

    # Convert star ratings to sentiment labels (1 = Positive, 0 = Negative)
    df['label'] = df['Score'].apply(lambda x: 1 if x >= 4 else 0)

    return df

# 🔹 Function to train the sentiment analysis model
@st.cache_resource
def train_model():
    # Load the dataset
    df = load_data()

    # Split dataset into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['label'], test_size=0.2, random_state=42)

    # Convert text data into numerical vectors using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train a Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)

    # Save the trained model and vectorizer
    joblib.dump(model, "sentiment_model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")

    return accuracy  # Return accuracy for display

# 🔹 Function to load the trained model
def load_model():
    model = joblib.load("sentiment_model.pkl")  # Load saved model
    vectorizer = joblib.load("vectorizer.pkl")  # Load vectorizer
    return model, vectorizer

# 🔹 Streamlit Web App Interface
def main():
    st.title("📢 Sentiment Analysis App")  # Title of the app
    st.write("Enter a review below to analyze its sentiment!")

    # Text input box for user to enter a review
    user_input = st.text_area("✍️ Enter your review:", "")

    if st.button("Analyze Sentiment"):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter a review!")
        else:
            # Load trained model & vectorizer
            model, vectorizer = load_model()

            # Transform user input into numerical format
            user_input_tfidf = vectorizer.transform([user_input])

            # Predict sentiment (0 = Negative, 1 = Positive)
            prediction = model.predict(user_input_tfidf)[0]

            # Display result
            if prediction == 1:
                st.success("😊 Positive Review!")
            else:
                st.error("😞 Negative Review!")

# Run the Streamlit app
if __name__ == "__main__":
    main()
