import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import json
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string

# --- SETUP ---
st.set_page_config(page_title="Intents Chatbot", page_icon="🤖")

# Show logo and header
col1,col2,col3,col4,col5 = st.columns(5)
with col3:
    st.image("intents_chatbot_logo.png", width=120)
st.markdown("""
    <div style='text-align: center;'>
        <h1>Welcome to Chatlee!</h1>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <hr>
    <div style='text-align: center;'>
        <h3>About Chatlee 🤖</h3>
        <p>
            <b>Chatlee</b> is a simple AI chatbot that responds based on your intents.<br>
            It can chat with you, tell jokes and more.<br>
            <b>Please keep the chats simple, it is a bot afterall</b><br>
            <i>
            Want to learn more about how it works? Ask me anything on 
            <a href="https://www.linkedin.com/in/khush-thakkar-b659b732b/" target="_blank">LinkedIn</a>
            </i>
        </p>
    </div>
    <hr>
""", unsafe_allow_html=True)

st.write("Type your message below and start chatting! ")

# --- LOAD MODEL AND ASSETS ---
model = tf.keras.models.load_model("intents_chatbot_2025.keras")
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    lbl_encoder = pickle.load(f)
with open('intents.json', 'r') as f:
    data = json.load(f)

# --- DOWNLOAD NLTK DATA ---
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# --- PREPROCESS FUNCTION ---
def preprocess(text):
    tokens = nltk.word_tokenize(text)
    tokens = [w.lower() for w in tokens]
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words and w not in string.punctuation]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

# --- SETUP SESSION STATE FOR CHAT HISTORY ---
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# --- DISPLAY CHAT HISTORY ---
for msg in st.session_state['messages']:
    if msg['role'] == 'user':
        st.markdown(f"<div style='text-align: right; color: #00c3ff'><b>You:</b> {msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='text-align: left; color: #00ff55'><b>Bot:</b> {msg['content']}</div>", unsafe_allow_html=True)

# --- USER INPUT BOX (changes key for every message for smooth experience) ---
user_input = st.text_input("You:", "", key=f"user_input_{len(st.session_state['messages'])}")

if user_input:
    # Add user message to history
    st.session_state['messages'].append({'role': 'user', 'content': user_input})

    # Preprocess and predict
    processed = preprocess(user_input)
    seq = tokenizer.texts_to_sequences([processed])
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    padded = pad_sequences(seq, maxlen=20, padding='post')

    pred = model.predict(padded)
    pred_class = pred.argmax(axis=1)[0]
    intent = lbl_encoder.inverse_transform([pred_class])[0]
    response = "Sorry, I don't understand that."
    for i in data['intents']:
        if i['tag'] == intent:
            response = np.random.choice(i['responses'])

    # Add bot response to history
    st.session_state['messages'].append({'role': 'bot', 'content': response})

    # Rerun to show new message (user and bot)
    st.rerun()
