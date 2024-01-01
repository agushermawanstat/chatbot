import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

logo_path = 'logokalbe.png'
st.image(logo_path, width=200)
df = pd.read_excel('Laptop tidak dapat terhubung ke Wi-Fi.xlsx')

# Tambahkan permintaan feedback di paling atas
feedback = st.slider("Berikan tingkat kepuasan Anda (1-100%):", 1, 100, 50)

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

    if len(similarities) == 0 or len(similarities) < top_k or all(similarity == 0 for similarity in similarities):
        # Check if detail question has already been asked
        if 'asked_detail_question' not in st.session_state:
            # Set flag to indicate that the detail question has been asked
            st.session_state.asked_detail_question = True
            # Tambahkan argumen key yang unik
            detail_question = st.text_area("To provide a more accurate answer, please provide details of your question or issue:")
            user_input += " " + detail_question
            return generate_response_tfidf_with_probability_and_detail(user_input, df)
    else:
        max_probability = max(similarities)
        if max_probability >= threshold_probability:
            top_k_indices = np.argsort(similarities)[-min(top_k, len(similarities)):][::-1]
            response_options = [(df['answer'].iloc[index], similarities[index]) for index in top_k_indices if index < len(df)]
            return response_options
        else:
            # Tambahkan argumen key yang unik
            user_input = st.text_area(f"Probabilitas jawaban tertinggi saat ini kurang dari {threshold_probability*100}%. Berikan lebih banyak detail pertanyaan atau masalah Anda:")
            return generate_response_tfidf_with_probability_and_detail(user_input, df)

# Inisialisasi variabel untuk melacak tingkat kepuasan dan jumlah respons
satisfaction_count = 0
satisfaction_ratings = []

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
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit UI
st.title("CIT-Knowledge Management Chatbot")

# Loop untuk menerima pertanyaan dari pengguna
while True:
    # Gantilah bagian while loop seperti di bawah agar sesuai dengan pola penggunaan Streamlit yang benar
    user_input = st.text_input("Enter your question (type 'exit' to exit):")
    if user_input.lower() == 'exit':
        break
    
    response_options = generate_response_tfidf_with_probability_and_detail(user_input, df)
    if response_options:
        for i, (response, probability) in enumerate(response_options, start=1):
            # Hitung gradasi warna sesuai dengan probabilitas
            if probability >= 0.8:
                color = "#ADFF2F"  # Hijau
            elif probability >= 0.5:
                color = "#FFD700"  # Kuning
            else:
                color = "#F08080"  # Merah
            
            # Tambahkan CSS untuk style kotak dengan gradasi warna yang lebih lembut (pastel)
            expander_id = f"expander_{i}"  # ID unik untuk setiap expander
            with st.expander(f"Option {i}: (Prob.: {probability:.0%}) {response.capitalize()}", expanded=False):
                # Gunakan widget `key` untuk mencegah DuplicateWidgetID
                satisfaction_rating = st.select_slider("Pilih tingkat kepuasan:", options=list(range(1, 101)), key=f"satisfaction_{i}")
                if satisfaction_rating:
                    satisfaction_count += 1
                    satisfaction_ratings.append(satisfaction_rating)

# Tampilkan visualisasi tingkat kepuasan dan jumlah respons
st.header("Feedback & Satisfaction Report")
st.write(f"Jumlah Respons: {satisfaction_count}")
if satisfaction_ratings:
    satisfaction_data = pd.DataFrame(satisfaction_ratings, columns=["Tingkat Kepuasan"])
    st.dataframe(satisfaction_data)
    # Visualisasi dalam bentuk diagram batang
    satisfaction_counts = satisfaction_data["Tingkat Kepuasan"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(satisfaction_counts.index, satisfaction_counts.values)
    ax.set_xlabel("Tingkat Kepuasan (%)")
    ax.set_ylabel("Jumlah Respons")
    ax.set_title("Distribusi Tingkat Kepuasan")
    st.pyplot(fig)
