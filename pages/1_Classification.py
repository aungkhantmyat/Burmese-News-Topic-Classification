import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import joblib
import pyidaungsu as pds
from sklearn.feature_extraction.text import TfidfVectorizer

#Load Stop Word File
stopwordslist = []
slist = []
with open("stopword.txt", encoding = 'utf8') as stopwordsfile:
    stopwords = stopwordsfile.readlines()
    slist.extend(stopwords)
    for w in range(len(slist)):
        temp = slist[w]
        stopwordslist.append(temp.rstrip())

def stop_word(sentence):
  new_sentence = []
  for word in sentence.split():
    if word not in stopwordslist:
      new_sentence.append(word)
  return(' '.join(new_sentence))

def tokenize(line):
    sentence = pds.tokenize(line,form="word")
    sentence = ' '.join([str(elem) for elem in sentence])
    sentence = stop_word(sentence)
    return sentence

# File Names
svm_model_filename = './Final_svm_model.pkl'
NB_model_filename = './Final_NB_model.pkl'
vectorizer_filename = './Final_vectorizer.pkl'
selector_filename= './Final_selector.pkl'
# filename1 = './svm_model.sav'
# filename2='./NB_model.sav'
# Load the SVM model
loaded_svm_model = joblib.load(svm_model_filename)
# Load the NB model
loaded_NB_model = joblib.load(NB_model_filename)
# Load the TfidfVectorizer
loaded_vectorizer = joblib.load(vectorizer_filename)
# Load the Selector
loaded_selector = joblib.load(selector_filename)

# # load the model from disk
# loaded_model1 = pickle.load(open(filename1, 'rb'))
# loaded_model2 = pickle.load(open(filename1, 'rb'))
#
# #Load Vectorizers from disk
# loaded_vectorizer = pickle.load(open("vectorizer_cap.pickle", "rb"))
# loaded_selector = pickle.load(open("selector_cap.pickle", "rb"))

with st.sidebar:
    selected = option_menu(
        menu_title="Topic Classification",
        options=["SVM", "Naive Bayes", "Evaluation"],
        icons=["1-circle-fill","2-circle-fill","bar-chart-line-fill"],
        menu_icon="diagram-3-fill",
        default_index=0
    )

if selected == "SVM":
    st.title(f'Myanmar News Classification System using {selected}')
    st.subheader("Input the News content below")
    sentence = st.text_area("Enter your news Content Here", height=200)
    sentence = tokenize(sentence)
    predict_btt = st.button("Predict")
    if predict_btt:
        data = loaded_selector.transform(loaded_vectorizer.transform([sentence]).toarray())
        prediction1 = loaded_svm_model.predict(data)

        if prediction1 == ['politic']:
            st.text("This is Politics News")
        elif prediction1 == ['crime']:
            st.text("This is Crime News")
        elif prediction1 == ['entertainment']:
            st.text("This is Entertainment News")
        elif prediction1 == ['business']:
            st.text("This is Business News")

if selected == "Naive Bayes":
    st.title(f'Myanmar News Classification System using {selected}')
    st.subheader("Input the News content below")
    sentence1 = st.text_area("Enter your news Content Here", height=200)
    sentence1 = tokenize(sentence1)
    predict_btt = st.button("Predict")
    if predict_btt:
        data = loaded_selector.transform(loaded_vectorizer.transform([sentence1]).toarray())
        prediction2 = loaded_NB_model.predict(data)

        if prediction2 == ['politic']:
            st.text("This is Politics News")
        elif prediction2 == ['crime']:
            st.text("This is Crime News")
        elif prediction2 == ['entertainment']:
            st.text("This is Entertainment News")
        elif prediction2 == ['business']:
            st.text("This is Business News")

if selected == "Evaluation":
    st.title(f"{selected} on Experimental Results")
    #Categorical Accuarcy data
    data = {
        'Category': ['Business', 'Crime', 'Entertainment', 'Politic'],
        'P (%) (NB)': [88, 83, 81, 84],
        'P (%) (SVM)': [87, 85, 77, 86],
        'R (%) (NB)': [82, 80, 90, 80],
        'R (%) (SVM)': [82, 77, 91, 78],
        'F1 (%) (NB)': [85, 82, 85, 82],
        'F1 (%) (SVM)': [84, 81, 84, 82]
     }
    # Convert data to DataFrame
    df = pd.DataFrame(data)
    # Convert DataFrame to HTML with index=False
    html_table = df.to_html(index=False)
    # Display the table
    st.header("**Experimental Results for Two Classifiers**")
    st.write(html_table, unsafe_allow_html=True)
    st.divider()
    # Define the different models accuarcy data
    bardata = {
        'Classifier': ['SVM', 'Naive Bayes', 'Random Forest','Decision Tree', 'KNN'],
        'Accuracy': [0.83, 0.84, 0.79, 0.69, 0.60]
    }
    # # Convert data to DataFrame
    df1 = pd.DataFrame(bardata)
    # Display the chart title
    st.header("**Accuracy Comparison with Different Models**")
    #plt.style.use('dark_background')
    fig, ax = plt.subplots()
    bars = ax.bar(df1['Classifier'], df1['Accuracy'], color='skyblue')
    # Add percentage values on bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.0%}', va='bottom')
    # Customize plot
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 1)  # Set y-axis limits to percentage scale (0 to 100%)
    # Rotate x-axis labels vertically
    plt.xticks(rotation=90)
    # Display the chart in Streamlit
    st.pyplot(fig)
