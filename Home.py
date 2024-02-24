import streamlit as st


def main():
    st.title("Enhancing NLP for Myanmar News Through Classification and Summarization")

    st.subheader("Current Challenges")
    st.markdown("""
    - Myanmar, with its rich cultural and linguistic diversity, presents a distinctive set of challenges for NLP tasks. In the context of Topic classification, implementing robust Topic Classification models can provide a structured framework for understanding and organizing the diverse range of topics prevalent in the Myanmar language which is still not well developed and utilized yet. This can aid in content organization and facilitate more targeted information retrieval for Myanmar News media and researchers.

    - As for summarization, the Myanmar texts often lengthy and intricate, benefit greatly from Text Summarization. Generating concise and coherent summaries not only facilitates quicker comprehension but also addresses the challenge of information overload. In domains like news articles, legal documents, and academic research, automatic summarization can significantly improve accessibility and information retrieval.
    """)

    st.subheader("Objective")
    st.markdown("""
    The primary goal of this research is to provide a better Myanmar Topic Classification and Summarization system to the users. The system also aims to contribute methodologies to Myanmar NLP research, particularly in the domain of news analysis and language-specific NLP applications to advance the state-of-the-art NLP for Myanmar language processing.

    The other objectives are:
    - To gather Myanmar News Article Corpus for research
    - To apply good stopwords to filter the corpus for further processing.
    - To apply the best vectorization technique for the Topic classification.
    - To evaluate and select only the important features for the Topic Classification using feature selection methods.
    - To deliver the best Topic Classification results using the best classifier model.
    - To deliver great accessible information that matters most to the users.
    Need to add for summarization.
    """)

    st.subheader("Main contents")
    st.subheader("Myanmar News Classification")
    st.markdown("""
    - In the Myanmar News Classification Part, we utilized the SVM and NB models to make predictions from the users.
      1. **Support Vector Machines (SVM)**: is a machine learning technique applicable for classifying textual data, with diverse uses in areas like credit risk assessment, medical diagnosis, text classification, and information extraction.  SVM is a large-margin classifier that finds a decision boundary between classes of data to separate them.  Users will be able to make predictions using the system's SVM model.
      2. **Naive Bayes (NB)**: a probabilistic classification algorithm which works on the principles of Bayes' theorem and is particularly suitable for problems involving text data with discrete features, such as word counts. Users will be able to make predictions using the system's NB model.
       """)

if __name__ == "__main__":
    main()
