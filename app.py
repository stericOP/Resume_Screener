import streamlit as st
import re
import pickle
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer



# Ensure you have downloaded the required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Loading models
try:
    clf = pickle.load(open("clf.pkl", 'rb'))
    tfidf = pickle.load(open("tfidf.pkl", 'rb'))
except Exception as e:
    st.error(f"Error loading models: {e}")

def cleanResume(resume_text):
    cleanText = re.sub("http\S+", " ", resume_text)
    cleanText = re.sub("@\S+", " ", cleanText)
    cleanText = re.sub("#\S+", " ", cleanText)
    cleanText = re.sub("RT|cc\S+", " ", cleanText)
    cleanText = re.sub('[%s]' % re.escape("""#$%&*+,-./:;<=>?@[\]^_`{|}~"""), " ", cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

# Web App
def main():
    st.title("Resume Screening App")
    st.write("Upload your resume to see the predicted category")
    uploaded_file = st.file_uploader("Upload Resume", type=["txt", "pdf"])

    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            resume_text = resume_bytes.decode("latin-1")

        # st.write("Original Resume Text:")
        # st.write(resume_text)
        
        cleaned_resume = cleanResume(resume_text)
        # st.write("Cleaned Resume Text:")
        # st.write(cleaned_resume)

        input_features = tfidf.transform([cleaned_resume])
        # st.write("Input Features:")
        # st.write(input_features)

        prediction_id = clf.predict(input_features)[0]  # Extract the first element
        # st.write("Prediction ID:")
        # st.write(prediction_id)

        category_mapping = {
            0: 'Advocate',
            1: 'Arts',
            2: 'Automation Testing',
            3: 'Blockchain',
            4: 'Business Analyst',
            5: 'Civil Engineer',
            6: 'Data Science',
            7: 'Database',
            8: 'DevOps Engineer',
            9: 'DotNet Developer',
            10: 'ETL Developer',
            11: 'Electrical Engineering',
            12: 'HR',
            13: 'Hadoop',
            14: 'Health and fitness',
            15: 'Java Developer',
            16: 'Mechanical Engineer',
            17: 'Network Security Engineer',
            18: 'Operations Manager',
            19: 'PMO',
            20: 'Python Developer',
            21: 'SAP Developer',
            22: 'Sales',
            23: 'Testing',
            24: 'Web Designing'
        }

        category_name = category_mapping.get(prediction_id, "Unknown")
        st.write("Predicted Category:")
        st.write(category_name)

if __name__ == "__main__":
    main()