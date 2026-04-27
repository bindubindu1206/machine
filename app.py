import streamlit as st
import pickle
import re

# Page config
st.set_page_config(page_title="Spam Detector", page_icon="📧", layout="wide")

# Load models
try:
    nb_model = pickle.load(open("nb_model.pkl", "rb"))
    lr_model = pickle.load(open("lr_model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
except:
    st.error("⚠️ Model files not found!")
    st.stop()

# Text cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    return text

# Prediction function
def predict(message, model):
    cleaned = clean_text(message)
    vector = vectorizer.transform([cleaned])
    pred = model.predict(vector)[0]
    prob = model.predict_proba(vector)[0]
    confidence = max(prob) * 100
    return pred, confidence, vector

# Sidebar
st.sidebar.title("⚙️ Settings")

model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["Naive Bayes", "Logistic Regression"]
)

mode = st.sidebar.radio(
    "Choose Mode",
    ["Prediction", "Learn Algorithm"]
)

st.sidebar.markdown("### 📊 About")
st.sidebar.info("""
Spam Detection using:
- TF-IDF
- Naive Bayes
- Logistic Regression
""")

# Select model
model = nb_model if model_choice == "Naive Bayes" else lr_model

# ===========================
# 🔮 PREDICTION MODE
# ===========================
if mode == "Prediction":

    st.title("📧 AI Spam Detection System")
    st.markdown("### 💬 Chat")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    user_input = st.text_input("Type your message")

    col1, col2 = st.columns([1,1])

    with col1:
        send = st.button("Send")
    with col2:
        clear = st.button("🧹 Clear Chat")

    if clear:
        st.session_state.messages = []

    if send:
        if user_input.strip() != "":
            # Store user message
            st.session_state.messages.append(("user", user_input))

            pred, confidence, vector = predict(user_input, model)

            if pred == 1:
                response = f"🚨 Spam ({confidence:.2f}%)"
            else:
                response = f"✅ Not Spam ({confidence:.2f}%)"

            # Store bot response
            st.session_state.messages.append(("bot", response))

    # Display chat
    for role, msg in st.session_state.messages:
        if role == "user":
            st.markdown(f"🧑 **You:** {msg}")
        else:
            st.markdown(f"🤖 **Bot:** {msg}")

    # Explanation
    if st.session_state.messages:
        last_user_msg = [m for m in st.session_state.messages if m[0] == "user"][-1][1]
        _, _, vector = predict(last_user_msg, model)

        st.markdown("### 🔍 Important Words")

        feature_names = vectorizer.get_feature_names_out()
        dense = vector.toarray()[0]

        important_words = [
            (feature_names[i], dense[i])
            for i in range(len(dense)) if dense[i] > 0
        ]

        important_words = sorted(important_words, key=lambda x: x[1], reverse=True)[:10]

        if important_words:
            for word, score in important_words:
                st.write(f"- {word}")
        else:
            st.write("No significant words found.")

# ===========================
# 🧠 LEARN MODE
# ===========================
elif mode == "Learn Algorithm":

    st.title("🧠 How the Model Works")

    st.subheader("📌 Naive Bayes")

    st.write("""
Naive Bayes is a probabilistic algorithm based on Bayes Theorem.

It assumes that each word in the message contributes independently to the probability of being spam or not spam.

Example:
Words like "free", "win", "money" increase spam probability.
""")

    st.latex(r"P(y|x) = \frac{P(x|y) P(y)}{P(x)}")

    st.write("""
👉 It calculates probability of a message being spam based on words present.
""")

    st.divider()

    st.subheader("📌 Logistic Regression")

    st.write("""
Logistic Regression is a linear model that assigns weights to each word.

It combines all words to calculate a final probability using a sigmoid function.
""")

    st.latex(r"P(y=1) = \frac{1}{1 + e^{-(w^T x + b)}}")

    st.write("""
👉 Words work together instead of independently.

👉 It usually gives better accuracy than Naive Bayes.
""")

    st.divider()

    st.subheader("⚖️ Key Difference")

    st.write("""
- Naive Bayes: Assumes words are independent  
- Logistic Regression: Learns relationships between words  

👉 Example:
"free" + "money" together strongly indicate spam (captured better by Logistic Regression)
""")