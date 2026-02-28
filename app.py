import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import pickle
from src.pipeline import Pipeline
from src.safety import safe_pipeline_run

API_KEY = st.secrets["OPENROUTER_API_KEY"]
MODEL = st.secrets["OPENROUTER_MODEL"]

@st.cache_resource
def load_pipeline():
    return Pipeline(
        index_path="data/rag/knowledge_base.index",
        kb_path="data/rag/knowledge_base.parquet",
        api_key=API_KEY,
        model=MODEL
    )

@st.cache_resource
def load_classifier():
    from huggingface_hub import hf_hub_download
    import pickle

    tokenizer = AutoTokenizer.from_pretrained("HavinChung/emotion-classifier")
    model = AutoModelForSequenceClassification.from_pretrained("HavinChung/emotion-classifier")
    
    pkl_path = hf_hub_download(repo_id="HavinChung/emotion-classifier", filename="label_encoder.pkl")
    with open(pkl_path, 'rb') as f:
        le = pickle.load(f)
    
    return tokenizer, model, le

def predict_emotion(text, tokenizer, model, le):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    pred = outputs.logits.argmax(dim=-1).item()
    return le.inverse_transform([pred])[0]

st.set_page_config(page_title="Emotional Support Chatbot")
st.title("Emotional Support Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

pipeline = load_pipeline()
tokenizer, classifier, le = load_classifier()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How are you feeling today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            emotion = predict_emotion(prompt, tokenizer, classifier, le)
            pipeline.conversation_history = st.session_state.messages[:-1]
            response, docs, status = pipeline.safe_run(prompt, emotion)
        st.markdown(response)
        if status != "crisis":
            st.caption(f"Detected emotion: {emotion}")

    st.session_state.messages.append({"role": "assistant", "content": response})