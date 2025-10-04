import streamlit as st
import pickle
from prediction import transform_text

# --- Load Model and Vectorizer ---
# Note the path to the models folder
try:
    tfidf = pickle.load(open('models/vectorizer.pkl', 'rb'))
    model = pickle.load(open('models/model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model files not found. Make sure 'vectorizer.pkl' and 'model.pkl' are in the 'models' folder.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model files: {e}")
    st.stop()

# --- Streamlit App Interface ---
st.set_page_config(layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 12px;
        padding: 10px 24px;
        font-size: 16px;
    }
    .stTextArea>div>div>textarea {
        background-color: #ffffff;
        border-radius: 8px;
    }
    h1, h2 {
        color: #333333;
    }
</style>
""", unsafe_allow_html=True)

st.title("✉️ Email & SMS Spam Classifier")
st.write("Enter a message below to check if it's spam or not.")

# Input text area
input_sms = st.text_area("Enter the message", height=150)

if st.button('Analyze Message'):
    if input_sms:
        # 1. Preprocess the input text using the function from prediction.py
        transformed_sms = transform_text(input_sms)

        # 2. Vectorize the preprocessed text
        vector_input = tfidf.transform([transformed_sms])

        # 3. Predict using the MNB model
        result = model.predict(vector_input)[0]

        # 4. Display the result
        if result == 1:
            st.header("🚨 Result: This looks like Spam")
        else:
            st.header("✅ Result: This seems like a legitimate message (Not Spam)")
    else:
        st.warning("Please enter a message to analyze.")
