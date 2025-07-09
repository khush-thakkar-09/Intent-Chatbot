# Intent-Chatbot

# Intents Chatbot (Chatlee) 🤖

Chatlee is a simple AI-powered chatbot built with TensorFlow, Keras, and Streamlit. It uses Natural Language Processing (NLP) to detect user intent and respond accordingly. This project is designed as a hands-on, practical introduction to building and deploying a basic intent-based chatbot.

---

## ✨ Features

- **Intent Detection:** Classifies user queries into pre-defined intents using a deep learning model.
- **Text Preprocessing:** Uses NLTK for tokenization, stopword removal, and lemmatization.
- **Pre-trained Embeddings:** Integrates GloVe word vectors for richer language understanding.
- **Interactive Chat UI:** Built with Streamlit for a clean, browser-based chat experience.
- **Session Chat History:** Remembers the full conversation in the browser session.
- **Easily Expandable:** Add more intents and responses by editing `intents.json`.
- **Simple, Fast, and Fun to Use!**

---

## 🚀 Quick Start

1. **Clone this repo or download all files.**
2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(If `requirements.txt` is missing, install: `streamlit tensorflow nltk numpy pickle5` and others as needed.)*
3. **Run the app:**
    ```bash
    streamlit run app.py
    ```
4. **Open your browser to:**  
   [http://localhost:8501](http://localhost:8501)
5. **Start chatting!**

---

## 🛠️ Files & Folders

- `app.py` - Streamlit web app code
- `intents.json` - Your intent patterns and responses
- `intents_chatbot_2025.keras` - Trained model
- `tokenizer.pkl` - Fitted tokenizer object
- `label_encoder.pkl` - Fitted label encoder
- `intents_chatbot_logo.png` - (Optional) Logo for your bot
- `README.md` - This file!

---

## 🧠 How It Works

- **Data Preparation:** All possible user inputs and bot responses are defined in `intents.json`.
- **NLP Preprocessing:** Each user message is tokenized, lowercased, cleaned, and lemmatized.
- **Model Prediction:** The message is vectorized and fed to a Keras LSTM model using GloVe embeddings.
- **Intent Matching:** The model predicts the intent tag, and a matching response is randomly chosen from the tag’s responses.
- **Persistent Chat:** Conversation history is stored in the browser with Streamlit’s `session_state`.

---

## 📈 Limitations

- Not a “true AI” chatbot—can only answer intents it’s trained on.
- Cannot answer open-domain or highly complex questions.
- Best used as a starting point for learning/development.

---

## 👤 Author

Built by [Khush Thakkar](https://www.linkedin.com/in/khush-thakkar-b659b732b/).

---

## 🙏 Acknowledgments

- [NLTK](https://www.nltk.org/) for text preprocessing
- [TensorFlow/Keras](https://www.tensorflow.org/) for model training
- [Streamlit](https://streamlit.io/) for deployment
- [GloVe](https://nlp.stanford.edu/projects/glove/) for embeddings

---

## 💡 How to Improve

- Add more intents and richer responses in `intents.json`.
- Integrate web APIs or database for dynamic responses.
- Replace LSTM with a transformer-based model.
- Deploy online with [Streamlit Cloud](https://streamlit.io/cloud) for public access.

---

*Pull requests and suggestions are welcome!*

