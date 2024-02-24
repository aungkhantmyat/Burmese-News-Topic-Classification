from nltk.cluster.util import cosine_distance
import re
import numpy as np
import pyidaungsu as pds
import networkx as nx
import streamlit as st

SENTENCE_SEPARATOR = [".", "?", "!", "။", "…", "\n"]

def sentence_tokenize(text):
  sentences = re.split("(?<=[" + "".join(SENTENCE_SEPARATOR) + "])\s*", text)
  if sentences[-1]:
    return sentences
  return sentences[:-1]


def read_text(text):
  sentences = sentence_tokenize(text)
  return sentences

def sentence_similarity(sent1, sent2):

  sent1 = [w.lower() for w in pds.tokenize(sent1, form='word')]
  sent2 = [w.lower() for w in pds.tokenize(sent2, form='word')]

  all_words = list(set(sent1 + sent2))

  vector1 = [0] * len(all_words)
  vector2 = [0] * len(all_words)

  for w in sent1:
    vector1[all_words.index(w)] += 1

  for w in sent2:
    vector2[all_words.index(w)] += 1

  if np.isnan(1 - cosine_distance(vector1, vector2)):
    return 0
  return 1 - cosine_distance(vector1, vector2)

def build_similarity_matrix(sentences):

  similarity_matrix = np.zeros((len(sentences),len(sentences)))

  for idx1 in range(len(sentences)):
    for idx2 in range(len(sentences)):
      if idx1!=idx2:
        similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1],sentences[idx2])
    return similarity_matrix

def generate_summary(text,top_n):
  summarize_text = []
  sentences = read_text(text)
  sentence_similarity_matrix = build_similarity_matrix(sentences)
  sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
  scores = nx.pagerank(sentence_similarity_graph)
  ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)),reverse=True)
  if not ranked_sentences:
        return ""  # Return empty summary and length 0
  else:
    for i in range(top_n):
      summarize_text.append(ranked_sentences[i][1].replace("\n", ""))
  return " ".join(summarize_text)

# on session start
st.set_page_config("NLP Summarization Demo", page_icon=":book:", layout="wide")

# layout ----------------
# title and desc
st.title("Extractive Myanmar News Summarization")
st.markdown(
        ""
    )
# input form area
st.header("Input Area")
input_form = st.empty()

# output area
col1, col2 = st.columns(2)
with col1:
      st.header("Full Input")
      full_input = st.empty()

with col2:
      st.header("Summarization")
      full_output = st.empty()

# interactivity -----------------
# set the input form depending on the sidebar input mode
with input_form.form("Input"):
      text_area = st.text_area("Your Text")
      topn = st.slider('No of sentences', 0, 10, 3)
      submitted = st.form_submit_button("Summarize!")
# run the model on submit
if submitted:
        text_of_input = text_area
        full_input.markdown(text_of_input)
        # pass input to model and print output
        with st.spinner("Summarizing..."):
            text_of_output = generate_summary(text_of_input,topn)
            full_output.markdown(text_of_output)
