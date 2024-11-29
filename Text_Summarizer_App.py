
import streamlit as st
from txtai.pipeline import Summary
from transformers import pipeline
from PyPDF2 import PdfReader
from rake_nltk import Rake
from textblob import TextBlob
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

nltk.download('stopwords')

@st.cache_resource
def summary_text(text):
    summary = Summary()
    result = summary(text)
    return result

def sentiment_analysis(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity, analysis.sentiment.subjectivity

def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as f:
        reader = PdfReader(f)
        page = reader.pages[0]
        text = page.extract_text()
    return text

def clean_text(text):
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    return cleaned_text.strip()

def generate_meaningful_topic(text, num_words=3):
    text = clean_text(text)
    vectorizer = CountVectorizer(stop_words='english', analyzer='word')
    doc_term_matrix = vectorizer.fit_transform([text])
    lda_model = LatentDirichletAllocation(n_components=1, random_state=42)
    lda_model.fit(doc_term_matrix)
    words = [vectorizer.get_feature_names_out()[i] for i in lda_model.components_[0].argsort()[-num_words:]]
    meaningful_topic = " ".join(words).capitalize()
    return meaningful_topic

def extract_top_keywords(text, top_n=5):
    r = Rake()
    r.extract_keywords_from_text(text)
    ranked_phrases = r.get_ranked_phrases_with_scores()
    top_keywords = sorted(ranked_phrases, key=lambda x: x[0], reverse=True)[:top_n]
    return top_keywords

st.set_page_config(layout="wide")
choice = st.sidebar.selectbox("Select your choice", ["Summarize Text", "Summarize Document"])

if choice == "Summarize Text":
    st.subheader("Summarize, Extract Keywords, and Analyze Sentiment")
    input_text = st.text_area("Enter your text here")
    if input_text and st.button("Process Text"):
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.markdown("**Your Input Text**")
            st.info(input_text, icon="ℹ️")
        with col2:
            st.markdown("**Summarized Text**")
            summarized_result = summary_text(input_text)
            st.success(summarized_result)
        with col2:
            st.markdown("**Top Extracted Keywords**")
            top_keywords = extract_top_keywords(summarized_result, top_n=5)
            for score, phrase in top_keywords:
                st.write(f"{phrase} (Score: {score})")
        with col3:
            st.markdown("**Sentiment Analysis**")
            polarity, subjectivity = sentiment_analysis(input_text)
            st.write(f"Polarity: {polarity}, Subjectivity: {subjectivity}")
        st.markdown("**Extracted Topics**")
        topics = generate_meaningful_topic(summarized_result)
        st.success(" ".join(topics))

elif choice == "Summarize Document":
    st.subheader("Summarize Document, Extract Keywords, and Analyze Sentiment")
    input_file = st.file_uploader("Upload your document", type=["pdf"])
    if input_file and st.button("Process Document"):
        with open("doc_file.pdf", 'wb') as f:
            f.write(input_file.getbuffer())
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.markdown("**Extracted Text from Document**")
            extracted_text = extract_text_from_pdf("doc_file.pdf")
            st.info(extracted_text)
        with col2:
            st.markdown("**Summarized Text**")
            summarized_result = summary_text(extracted_text)
            st.success(summarized_result)
        with col2:
            st.markdown("**Top Extracted Keywords**")
            top_keywords = extract_top_keywords(summarized_result, top_n=5)
            for score, phrase in top_keywords:
                st.write(f"{phrase} (Score: {score})")
        with col3:
            st.markdown("**Sentiment Analysis**")
            polarity, subjectivity = sentiment_analysis(extracted_text)
            st.write(f"Polarity: {polarity}, Subjectivity: {subjectivity}")
        st.markdown("**Extracted Topics**")
        topics = generate_meaningful_topic(summarized_result)
        st.success(" ".join(topics))
