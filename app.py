import streamlit as st
import matplotlib.pyplot as plt
from sklearn .feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag


nltk.download("punkt_tab") #nltk res down
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger_eng")


st.set_page_config(page_title="Resume Job Match Scorer",page_icon="📑",layout="wide") #page setup

st.markdown("""
Upload your resume (PDF) and paste a job description to see how well they match!  
This tool uses **TF-IDF + Cosine Similarity** to analyze your resume against job requirements.
""")

with st.sidebar:
    st.header("About")
    st.info("""
    This tool helps you:
    - Measures how your resume matches a job description
    - Identify important job keywords
    - Improve your reseume based on missing terms
    """)
    st.header("How It works")
    st.write("""
    1. Upload your resume (PDF)
    2. Paste the job description
    3. Click **Analyze Match**
    4. Review score & suggetion
    """)


def extract_text_from_pdf(uploaded_file): #help funct
    try:
        pdf_reder=PyPDF2.PdfReader(uploaded_file)
        text=""
        for page in pdf_reder.pages:
            text=text+page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF:{e}")
        return ""
    
    
def clean_text(text):
    text=text.lower()
    text=re.sub(r'[^a-zA-Z\s]','',text)
    text=re.sub(r'\s+',' ',text).strip()
    return text


def  remove_stopwords(text):
    stop_words=set(stopwords.words('english'))
    words=word_tokenize(text)
    return " ".join([word for word in words if word not in stop_words])


def calculate_similarity(resume_text,job_description):
    resume_processed=remove_stopwords(clean_text(resume_text))
    job_processed=remove_stopwords(clean_text(job_description))
    vectorizer=TfidfVectorizer()
    tfidf_matrix=vectorizer.fit_transform([resume_processed,job_processed])
    score=cosine_similarity(tfidf_matrix[0:1],tfidf_matrix[1:2])[0][0]*100
    return round(score,2),resume_processed,job_processed

def main(): #main fun

    uploaded_file=st.file_uploader("Upload your resuem (PDF)",type=['pdf'])
    job_description=st.text_area("Paste the job description",height=200)

    if st.button("Analyze Match"):
        if not uploaded_file:
            st.warning("Please upload your resume")
            return
        if not job_description:
            st.warning("Please paste the job description")
            return
        
        
        with st.spinner("Analyzing your resume...."):
            resume_text=extract_text_from_pdf(uploaded_file)
            if not resume_text:
                st.error("could not extract text from pdf. please try another pdf")
                return 
            
            similarity_score,resume_processed,job_processed=calculate_similarity(resume_text,job_description)

            st.subheader("Results")
            st.metric("Match Score",f"{similarity_score:.2f}%")

            fig,ax=plt.subplots(figsize=(6,0.5)) #chart
            colors=['#ff4b4b','#ffa726','#0f9d58']
            color_index=min(int(similarity_score//33),2)
            ax.barh([0],[similarity_score],color=colors[color_index])
            ax.set_xlim(0,100)
            ax.set_xlabel("Match percentage")
            ax.set_yticks([])
            ax.set_title("Resume Job Match")
            st.pyplot(fig)

            if similarity_score<40:
                st.warning("Low Match, consider tailoring your resume more closely.")
            elif similarity_score<70:
                st.info("Good Match. Your resume aligin fairly well")
            else:
                st.success("Excellent Match ! Your resume strongly aligns.")

if __name__=="__main__":
    main()
