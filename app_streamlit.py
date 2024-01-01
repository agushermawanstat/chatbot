import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set background color
st.markdown(
    """
    <style>
        body {
            background-color: #ffffff;
        }
        .custom-warning {
            background-color: #4CAF50;  /* Green background color */
            color: white; /* White text color */
            padding: 10px; /* Add some padding */
            margin: 10px 0; /* Add some margin */
        }
        .response-box {
            border-radius: 15px;
            padding: 10px;
            margin: 10px 0;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Load data
logo_path = 'logokalbe.png'
st.image(logo_path, width=200)
df = pd.read_excel('Laptop tidak dapat terhubung ke Wi-Fi.xlsx')

# Train LSTM model (Let's use st.cache for caching the model)
@st.cache(allow_output_mutation=True)
def train_lstm_model():
    train_data = df["question"].tolist() + df["answer"].tolist()
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(train_data)
    total_words = len(tokenizer.word_index) + 1

    input_sequences = []
    for line in train_data:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    max_sequence_length = max([len(x) for x in input_sequences])
    input_sequences = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')

    X, y = input_sequences[:, :-1], input_sequences[:, -1]
    y = tf.keras.utils.to_categorical(y, num_classes=total_words)

    model = Sequential()
    model.add(Embedding(total_words, 50, input_length=max_sequence_length-1))
    model.add(LSTM(100))
    model.add(Dense(total_words, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), verbose=1)

    return model

model = train_lstm_model()

# Chatbot function
def generate_response_tfidf_with_probability_and_detail(user_input, df, top_k=5, threshold_probability=0.25):
    vectorizer = TfidfVectorizer()
    corpus = df['question'].tolist() + df['answer'].tolist()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    user_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()

    max_probability = max(similarities)
    if max_probability >= threshold_probability:
        top_k_indices = np.argsort(similarities)[-min(top_k, len(similarities)):][::-1]
        response_options = [(df['answer'].iloc[index], similarities[index]) for index in top_k_indices if index < len(df)]
        return response_options
    else:
        # Custom warning message
        st.markdown(
            """
            <div class="custom-warning">
                Kindly provide a comprehensive and detailed description of the issue you are facing, and I will offer the solution as accurately as possible!
            </div>
            """,
            unsafe_allow_html=True
        )

# Streamlit UI
st.title("CIT-Knowledge Management Chatbot")

# Get user input with wider input box and the same prompt as before
user_input = st.text_area("Enter your question (type 'exit' to exit):", key='user_input')
st.markdown(
    """
    <style>
        .input-container {
            position: relative;
        }
        .submit-button {
            background-color: #4CAF50; /* Green background color */
            color: white; /* White text color */
            padding: 10px 20px; /* Add padding to the button */
            border: none; /* Remove button border */
            border-radius: 5px; /* Add border radius to the button */
            font-size: 16px; /* Adjust font size as needed */
            position: absolute;
            bottom: 10px;
            right: 10px;
        }
        textarea {
            width: 100%; /* Set the width of the textarea to 100% of the container */
            border: none; /* Remove the textarea border */
            padding: 10px; /* Add padding for a better appearance */
            font-size: 16px; /* Adjust font size as needed */
        }
    </style>
    """,
    unsafe_allow_html=True
)

if user_input.lower() != 'exit':
    response_options = generate_response_tfidf_with_probability_and_detail(user_input, df)
    if response_options:
        for i, (response, probability) in enumerate(response_options, start=1):
            # Define response box color based on probability
            if probability >= 0.8:
                color = "#ADFF2F"  # Green
            elif probability >= 0.5:
                color = "#FFD700"  # Yellow
            else:
                color = "#F08080"  # Red

            # Display response with colored box
            st.markdown(
                f"""
                <div class="response-box" style="background-color: {color};">
                    Option {i}: (Prob.: {probability:.0%}) {response.capitalize()}
                </div>
                """,
                unsafe_allow_html=True
            )
