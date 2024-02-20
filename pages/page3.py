import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import random
from sklearn import svm, metrics
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
import pyidaungsu as pds

dataset = "articles.csv"
data = pd.read_csv(dataset)

colslist = ['Index','News', 'Category', 'category_id']
data.columns = colslist

list_of_dicts = data.to_dict(orient='records')

final_dict_list=[]
for obj in list_of_dicts:
    text = obj['News']
    sentences = text.split("။")
    # Remove any empty strings resulting from consecutive delimiters
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    my_dict_list = [{"Index":obj['Index'],"News": item, "Category":obj['Category'],"category_id":obj['category_id']} for item in sentences]
    final_dict_list.extend(my_dict_list)
data_list = [{'Index': i+1, 'News': item['News'], 'Category': item['Category'], 'category_id': item['category_id']} for i, item in enumerate(final_dict_list)]

data = pd.DataFrame(data_list)

stopwordslist = []
slist = []
with open("stopwords.txt", encoding = 'utf8') as stopwordsfile:
    stopwords = stopwordsfile.readlines()
    slist.extend(stopwords)
    for w in range(len(slist)):
        temp = slist[w]
        stopwordslist.append(temp.rstrip())


# import pyidaungsu as pds
import re

# CleanPattern = re.compile(r'\d+|[၊။!-/:_\'’"”■—[-`{-~\t ]|[A-Za-z0-9]')
CleanPattern = re.compile(r'\d+|[၊။!-/:_\'’"”■—[-`{-~\t ]')
def clean_sentence(sentence):
     # sentence = sentence.replace("_"," ")
     sent = CleanPattern.sub(" ",str(sentence))
     return sent

# Removed everything from the stopword list.
def stop_word(sentence):
  new_sentence = []
  for word in sentence.split():
    if word not in stopwordslist:
      new_sentence.append(word)
  return(' '.join(new_sentence))


def tokenize(line): # Changing it into one word structure.
    sentence = clean_sentence(line)
    sentence = pds.tokenize(sentence,form="word")
    sentence = ' '.join([str(elem) for elem in sentence])
    sentence = stop_word(sentence)
    return sentence

data['News'] = data['News'].apply(tokenize)

# One or more digits.
# Specific Burmese characters (၊။).
# Various special characters.
# Uppercase and lowercase letters.


# Assuming your dataset has columns News,Category
features = data['News']
labels = data['Category']

train_data_list = []
test_data_list = []

# Iterate over each unique class and split the data
for category in labels.unique():
    # Filter data for the current category
    category_data = data[data['Category'] == category]

    # Split the data into train and test sets (80% train, 20% test)
    train_set, test_set = train_test_split(category_data, test_size=0.2, random_state=42)

    # Append the train and test sets to the respective lists
    train_data_list.append(train_set)
    test_data_list.append(test_set)

# Create DataFrames from the lists
train_data = pd.concat(train_data_list, ignore_index=True)
test_data = pd.concat(test_data_list, ignore_index=True)


# For Training

train_data = train_data[["News", "Category"]]

X_train = np.array(train_data["News"])
y_train = np.array(train_data["Category"])

# For Testing

test_data = test_data[["News", "Category"]]

X_test = np.array(test_data["News"])
y_test = np.array(test_data["Category"])


# def tokenize(line):
#     sentence = pds.tokenize(line,form="word")
#     return sentence

vectorizer = TfidfVectorizer(tokenizer=tokenize,ngram_range=(1,2))
# X_train = vectorizer.fit_transform(X_train)
# X_test = vectorizer.fit_transform(X_test)

X_train = vectorizer.fit_transform(X_train)  # Fit and transform on the training data
X_test = vectorizer.transform(X_test)  # Transform using the same vectorizer on the test data


# Import classifiers and performance metrics


# linear kernel model
svm_model = svm.SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Naive Bayes Model
NB_model = MultinomialNB()
NB_model.fit(X_train,y_train)

# predict
y_pred = svm_model.predict(X_test)

# Main Part

import joblib

# Save the SVM model
svm_model_filename = 'svm_model'+str(random.randint(1,4000))+'.pkl'
joblib.dump(svm_model, svm_model_filename)

# Save the NB model
NB_model_filename = 'NB_model'+str(random.randint(1,4000))+'.pkl'
joblib.dump(NB_model, NB_model_filename)

# Save the TfidfVectorizer
vectorizer_filename = 'vectorizer'+str(random.randint(1,4000))+'.pkl'
joblib.dump(vectorizer, vectorizer_filename)


# Load the SVM model
loaded_svm_model = joblib.load(svm_model_filename)

# Load the NB model
loaded_NB_model = joblib.load(NB_model_filename)

# Load the TfidfVectorizer
loaded_vectorizer = joblib.load(vectorizer_filename)




# Load CSV into DataFrame
df = pd.read_csv('TestingCase.csv')

# Initialize lists to store predictions and actual classes
predictions = []
actual_classes = []

# Iterate through each row in the DataFrame
for index, row in df.iterrows():
    user_input = tokenize(row['news'])  # Use the news column as input
    data = loaded_vectorizer.transform([user_input]).toarray()
    output = loaded_svm_model.predict(data)
    predictions.append(output[0])  # Append the prediction
    actual_classes.append(row['class'])  # Append the actual class

# Calculate accuracy for each category
categories = df['class'].unique()
category_accuracies = {}
for category in categories:
    indices = [i for i, x in enumerate(actual_classes) if x == category]
    category_predictions = [predictions[i] for i in indices]
    category_actual = [actual_classes[i] for i in indices]
    category_accuracy = accuracy_score(category_actual, category_predictions)
    category_accuracies[category] = (category_accuracy, len(indices))

# Print accuracies
st.write("Accuracy for each category (SVM):")
for category, (accuracy, count) in category_accuracies.items():
    st.write(f"{category} = {round(accuracy*100)}% ({round(accuracy*25)}/25)")

# Iterate through each row in the DataFrame
for index, row in df.iterrows():
    user_input = tokenize(row['news'])  # Use the news column as input
    data = loaded_vectorizer.transform([user_input]).toarray()
    output = loaded_NB_model.predict(data)
    predictions.append(output[0])  # Append the prediction
    actual_classes.append(row['class'])  # Append the actual class

# Calculate accuracy for each category
categories = df['class'].unique()
category_accuracies = {}
for category in categories:
    indices = [i for i, x in enumerate(actual_classes) if x == category]
    category_predictions = [predictions[i] for i in indices]
    category_actual = [actual_classes[i] for i in indices]
    category_accuracy = accuracy_score(category_actual, category_predictions)
    category_accuracies[category] = (category_accuracy, len(indices))

# Print accuracies
st.write("Accuracy for each category (NB):")
for category, (accuracy, count) in category_accuracies.items():
    st.write(f"{category} = {round(accuracy*100)}% ({round(accuracy*25)}/25)")

st.divider()

    def get_vectorizer_content():
        # Load your vectorizer file
        with open(vectorizer_filename, 'rb') as file:
            vectorizer_content = file.read()
        return vectorizer_content

    def get_svm_model_content():
        # Load your model file
        with open(svm_model_filename, 'rb') as file:
            model_content = file.read()
        return model_content
    
    def get_NB_model_content():
        # Load your model file
        with open(NB_model_filename, 'rb') as file:
            model_content = file.read()
        return model_content

    # Create a button to trigger the download of the vectorizer file
    vectorizer_button = st.download_button(label="Download Vectorizer", data=get_vectorizer_content(), file_name=vectorizer_filename, mime='application/octet-stream', key="vectorizer_btn")

    # Create a button to trigger the download of the model file
    SVM_model_button = st.download_button(label="Download Model", data=get_svm_model_content(), file_name=svm_model_filename, mime='application/octet-stream', key="SVM_model_Btn")
    # Create a button to trigger the download of the model file
    NB_model_button = st.download_button(label="Download NB Model", data=get_svm_model_content(), file_name= NB_model_filename, mime='application/octet-stream', key="NB_model_Btn")


