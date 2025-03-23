import streamlit as st
import fitz  # PyMuPDF
import re
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure Page
st.set_page_config(page_title="Resume Screening", page_icon="📄", layout="wide")

# Function to extract text from PDF resumes
def extract_text_from_pdf(uploaded_file):
    try:
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            text = "".join(page.get_text() for page in doc)
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

# Function to extract candidate details
def extract_candidate_details(text):
    email_match = re.search(r"[a-zA-Z0-9+_.-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]+", text)
    email = email_match.group() if email_match else "Not found"
    phone_match = re.search(r"\b\d{10}\b", text)
    phone = phone_match.group() if phone_match else "Not found"
    words = text.split()
    candidate_name = " ".join(words[:2]) if len(words) > 1 else "Not found"
    return candidate_name, email, phone

# Function to generate CSV
def generate_csv(data):
    df = pd.DataFrame(data, columns=["Candidate Name", "Email", "Mobile Number", "Score"])
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()

# Sidebar for Navigation
st.sidebar.title("🔍 Resume Screening System")
page = st.sidebar.radio("Select Page", ["📤 Upload Resumes", "✍️ Job Description & Results"])

if page == "📤 Upload Resumes":
    st.title("📤 Upload Resume PDFs")
    uploaded_files = st.file_uploader("Upload resumes", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        st.session_state["uploaded_files"] = uploaded_files
        st.success("Resumes uploaded successfully!")

elif page == "✍️ Job Description & Results":
    st.title("✍️ Enter Job Description")
    job_desc = st.text_area("Paste job description here", height=150)
    uploaded_files = st.session_state.get("uploaded_files", [])
    
    if st.button("📊 Process Resumes"):
        if uploaded_files and job_desc:
            resume_texts = [extract_text_from_pdf(file) for file in uploaded_files]
            
            # TF-IDF Vectorization
            vectorizer = TfidfVectorizer(stop_words="english")
            vectors = vectorizer.fit_transform([job_desc] + resume_texts)
            
            # Compute Cosine Similarity
            similarity_scores = cosine_similarity(vectors[0], vectors[1:])[0]
            
            candidate_names = [file.name for file in uploaded_files]
            candidates_data = []
            
            for i, (name, text, score) in enumerate(zip(candidate_names, resume_texts, similarity_scores)):
                candidate_name, email, phone = extract_candidate_details(text)
                candidates_data.append([candidate_name, email, phone, f"{score:.2f}"])
            
            # 📥 CSV Download Link
            csv_data = generate_csv(candidates_data)
            st.download_button("📥 Download Candidates CSV", data=csv_data, file_name="candidates_scores.csv", mime="text/csv")
            
            # 📊 Bar Chart Visualization
            st.subheader("📊 Candidate Score Comparison")
            df = pd.DataFrame({"Candidate Name": candidate_names, "Score": similarity_scores})
            fig = px.bar(df, x="Candidate Name", y="Score", text="Score", color="Score", color_continuous_scale="blues")
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            st.plotly_chart(fig)
            
            # Winning Candidate
            winner_index = np.argmax(similarity_scores)
            winning_resume_text = resume_texts[winner_index]
            winner_name, winner_email, winner_phone = extract_candidate_details(winning_resume_text)
            
            st.session_state["winning_resume_text"] = winning_resume_text
            st.session_state["winner_name"] = winner_name
            
            st.subheader("🏆 Winning Candidate Details")
            st.write(f"**Candidate Name:** {winner_name}")
            st.write(f"**Email:** {winner_email}")
            st.write(f"**Mobile Number:** {winner_phone}")
    
    if "winner_name" in st.session_state and st.button(f"📜 {st.session_state['winner_name']}'s Resume"):
        st.session_state["show_winner_resume"] = True
    
    if st.session_state.get("show_winner_resume", False):
        st.text_area("Resume Text:", st.session_state["winning_resume_text"], height=300)
