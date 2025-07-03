import streamlit as st
import joblib
import pandas as pd

spam_model=joblib.load("spam_classifier.pkl")
language_model=joblib.load("lang_det.pkl")
news_model=joblib.load("news_cat.pkl")
review_model=joblib.load("review.pkl")

# Page config
st.set_page_config(page_title="LENS eXpert (NLP Suites)", layout="wide", page_icon="🤖")

# Custom CSS to shrink sidebar
st.markdown("""
    <style>
        /* Shrink sidebar width */
        [data-testid="stSidebar"] {
            width: 220px;
        }

        [data-testid="stSidebar"] > div:first-child {
            width: 220px;
        }

        /* Make body background light if needed */
        .main {
            background-color: #f9f9f9;
        }
    </style>
""", unsafe_allow_html=True)

# ✅ TOP BANNER (main area — not sidebar)
st.markdown("""
    <div style='background: linear-gradient(to right, #1e3c72, #2a5298); 
                padding: 20px 40px; 
                border-radius: 15px; 
                color: white; 
                font-size: 24px; 
                font-weight: bold; 
                box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.2); 
                margin-bottom: 20px; 
                text-align: center;'>
       🤖LENS eXpert (NLP Suites)
    </div>
""", unsafe_allow_html=True)

tab1,tab2,tab3,tab4=st.tabs(["📩Spam Classifier","⭐Food Sentiments Reviews","🌍Language Detection","📢News Classification"])

with tab1:
    st.header("🚫  Welcome To The Spam Classifier 📩  ")
    msg=st.text_input("✍️Enter Message",key="input1")
    if st.button("🔍Detect Spam",key="btn1"):
        pred=spam_model.predict([msg])
        if pred[0]==0:
            st.warning("🚫 Spam (Unwanted/Scam Message)")
        else:
            st.success("📩 Not Spam (Genuine Message)")

 
    uploaded_file = st.file_uploader("📁 Upload a file (CSV/TXT):",type=["csv", "txt"])
   
  
    if uploaded_file:
            
        df_spam=pd.read_csv(uploaded_file,header=None,names=['Msg'])
       
        pred=spam_model.predict(df_spam.Msg)
        df_spam.index=range(1,df_spam.shape[0]+1)
        df_spam["Prediction"]=pred
        df_spam["Prediction"]=df_spam["Prediction"].map({0:'Spam',1:'Not Spam'})
        st.dataframe(df_spam)


with tab2:
    st.header("🍽️ Welcome To Food Sentiments Reviews ")
    msg=st.text_input("✍️Enter Review",key="input2")
    if st.button("🔍Analyze Review",key="btn2"):
        pred=review_model.predict([msg])
        if pred[0]==0:
            st.warning("👎 – Disliked Food😔")
        else:
            st.success("👍 – Liked Food😃")
            st.balloons()

    uploaded_file = st.file_uploader("📁 Upload a file (CSV/TXT):",type=["csv", "txt"],key="load2")
   
  
    if uploaded_file:
            
        df_review=pd.read_csv(uploaded_file,header=None,names=['Msg'])
       
        pred=review_model.predict(df_review.Msg)
        df_review.index=range(1,df_review.shape[0]+1)
        df_review["Prediction"]=pred
        df_review["Prediction"]=df_review["Prediction"].map({0:'Dislike',1:'Like'})
        st.dataframe(df_review)


with tab3:
    st.header("🌐 Welcome To Language Detection ")
    msg=st.text_input("✍️Enter a sentence",key="input3")
    if st.button("🚀Predict Language",key="btn3"):
        pred=language_model.predict([msg])
        st.markdown(f"<h3 style='color: green;'>🌍Detected Language: {pred}</h3>", unsafe_allow_html=True)

  
    uploaded_file = st.file_uploader("📁 Upload a file (CSV/TXT):",type=["csv", "txt"],key="load3")
   
  
    if uploaded_file:
            
        df_lang=pd.read_csv(uploaded_file,header=None,names=['Msg'])
       
        pred=language_model.predict(df_lang.Msg)
        df_lang.index=range(1,df_lang.shape[0]+1)
        df_lang["Prediction"]=pred
        df_lang["Prediction"]=df_lang["Prediction"]
        st.dataframe(df_lang)


with tab4:
    st.header("🗞️ Welcome To News Classification")
    msg=st.text_input("✍️Enter Message",key="input4")
    if st.button("🔍News Analyze",key="btn4"):
        pred=news_model.predict([msg])
        #st.markdown(f"<h3 style='color: green;'>📢Detected News: {pred}</h3>", unsafe_allow_html=True)
        st.success(pred)

    
    uploaded_file = st.file_uploader("📁 Upload a file (CSV/TXT):",type=["csv", "txt"],key="load4")
   
  
    if uploaded_file:
            
        df_news=pd.read_csv(uploaded_file,header=None,names=['Msg'])
       
        pred=news_model.predict(df_news.Msg)
        df_news.index=range(1,df_news.shape[0]+1)
        df_news["Prediction"]=pred
        df_news["Prediction"]=df_news["Prediction"]
        st.dataframe(df_news)


# Sidebar
with st.sidebar:
    st.markdown("## 👋 **Hello User, Welcome To**")
    st.markdown("###  **🤖 LENS eXpert Model**")

st.sidebar.image("C:/Users/prave/Desktop/Gen AI/NLP Videos/NLP Project Programs/Model.jpeg")

with st.sidebar.expander("ℹ️ About Us"):
    st.write("""We are group of students trying to understand the concept of NLP Models
            
This tool is created to help users easily analyze and understand data using AI models.
We aim to make your experience simple and effective.
             
💡 Technologies: Python, Streamlit, Pandas, Scikit-learn, nltk, etc.

🔍 ML Models: Trained on real-world text datasets""")

with st.sidebar.expander("📞 Contact Us"):
    st.write("📱+91 6387583178")
    st.write("""
For any questions, feedback, or support,
please reach out at: rishaloopandit020@gmail.com""")
    

with st.sidebar.expander("🤝Help"):
    st.write("If you have trouble using the model, check the instructions or 📞contact us.")

    st.markdown("---")  # Horizontal line

st.markdown("""
<div style='text-align: center; font-size: 16px;'>
🔍 <b>LENS eXpert</b> | Built with ❤️ using <b>Python</b>, <b>Streamlit</b>, <b>Pandas</b>,  <b>sklearn</b>, and, <b>nltk Library</b><br>
📚 For Learning & Academic Use | © 2025
</div>
""", unsafe_allow_html=True)
