import gensim, re
import numpy as np
import pandas as pd
import pickle
from os import listdir

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import sys
import os

from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Embedding

from underthesea import word_tokenize

sep = os.sep # directory separtor
data_folder = "data" # folder that contains data and model
data_file = "Data.csv"
model_version = "7"

enable_train_new_model = True # Set this to False if you want to reuse an existing model with model_version

# Dictionary to scale the dataset for a more balanced dataset
freq = dict({("#tìmngườiyêu", 4), ("#lcd", 9), ("#gópý", 11), ("#bócphốt", 12), ("#hỏiđáp", 13), ("#tìmbạn", 23), ("#tâmsự", 1), ("#chiasẻ", 1)})

def loadDataFromCSV():
    df = pd.read_csv(data_folder + sep + data_file)
    df['tag'] = df['tag'].fillna("#LCD")
    return df

# Create a tokenizer to use later on
def txtTokenizer(texts):
    # texts: A list of sentences
    tokenizer = Tokenizer()
    # fit the tokenizer on our text
    tokenizer.fit_on_texts(texts)

    # get all words that the tokenizer knows
    word_index = tokenizer.word_index
    return tokenizer, word_index

# Remove trash symbols and spaces + Lower the case of all data
def preProcess(sentences):
    # sentences: The list of all sentences in a confession
    text = [word_tokenize(sentence, format="text") for sentence in sentences]
#     text = [re.sub(r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))", '', sentence) for sentence in sentences if sentence!='']
    text = [re.sub(r'([^\s\w]|"\')+', '', sentence) for sentence in text if sentence!='']
    text = [sentence.lower().strip().split() for sentence in text]
    return text

# Pre-process tags to become lowercase and standardize them into 8 categories
def preProcessTag(tag):
    temp = tag.lower().replace(" ", "")
    if "ngườiyêu" in temp:
        return "#tìmngườiyêu"
    elif "tâmsự" in temp:
        return "#tâmsự"
    elif "gópý" in temp:
        return "#gópý"
    elif "bócphốt" in temp:
        return "#bócphốt"
    elif "hỏiđáp" in temp:
        return "#hỏiđáp"
    elif "bạn" in temp or "info" in temp or "ngườiđichơi" in temp:
        return "#tìmbạn"
    elif "chiasẻ" in temp:
        return "#chiasẻ"
    elif "lcd" in temp:
        return "#lcd"
    else:
        return "error"

# load the data from the dataframe and do pre-processing to the sentences
# in each confessions as well as labelling them
def loadData(df):
    texts = []
    labels = []
    for sample in df['content']:
        sentences = sample.split('.')
        sentences = preProcess(sentences)
        tag = df.loc[df.content == sample, 'tag'].values[0]
        label = [preProcessTag(tag) for _ in sentences]
        # [tag tag tag tag tag]
        for i in range(freq[preProcessTag(tag)]):
            texts = texts + sentences
            labels = labels + label
    return texts, labels

if __name__ == '__main__':
    dataframe = loadDataFromCSV()
    texts, labels = loadData(dataframe)
    tokenizer, word_index = txtTokenizer(texts)

    # put the tokens in a matrix
    X = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(X, maxlen=500)

    # prepare the labels
    Y = pd.get_dummies(labels)

    if enable_train_new_model:
        # Save the new model
        file = open(data_folder + sep + "data_" + model_version + ".pkl", 'wb')
        pickle.dump([X,Y, texts],file)
        file.close()
    else:
        file = open(data_folder + sep + "data_" + model_version + ".pkl", 'rb')
        X, Y, texts = pickle.load(file)
        file.close()

    # Split train an test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, shuffle=True)

    # train Word2Vec model on our data
    if enable_train_new_model:
        word_model = gensim.models.Word2Vec(texts, vector_size=300, min_count=1, epochs=10)
        word_model.save(data_folder + sep + "word_model_" + model_version + ".save")

        embedding_matrix = np.zeros((len(word_model.wv) + 1, 300))
        for i, vec in enumerate(word_model.wv.vectors):
            embedding_matrix[i] = vec

        model = Sequential()
        model.add(Embedding(len(word_model.wv)+1,300,input_length=X.shape[1],weights=[embedding_matrix],trainable=False))
        model.add(LSTM(300,return_sequences=False))
        model.add(Dense(Y.shape[1],activation="softmax"))
        model.summary()
        model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=['acc'])

        batch = 64
        epochs = 1
        model.fit(X_train,Y_train,batch,epochs)
        model.save(data_folder + sep + "predict_model_" + model_version + ".save")
    else:
        word_model = gensim.models.Word2Vec.load(data_folder + sep + "word_model_" + model_version + ".save")
        model = load_model(data_folder + sep + "predict_model_" + model_version + ".save")
        model.summary()

    model.evaluate(X_test, Y_test)

    # Test model
    file = open(data_folder + sep + "My_data.txt", "r", encoding="utf8")
    input_string = file.read().split('.')
    X_dev = tokenizer.texts_to_sequences(preProcess(input_string))
    print(tokenizer.sequences_to_texts(X_dev))
    X_dev = pad_sequences(X_dev, maxlen=len(X_test[0]))
    print("Predicting...")
    result_prediction_dict = dict()
    prediction_cus = model.predict(X_dev, verbose=0)
    print(tokenizer.sequences_to_texts(X_dev))
    for i in range(len(prediction_cus)):
        result_tag = Y_train.columns[np.argmax(prediction_cus[i])]
        # print(result_tag)
        result_prediction_dict[result_tag] = result_prediction_dict.get(result_tag, 0) + 1
    print(result_prediction_dict)
    print(max(zip(result_prediction_dict.values(), result_prediction_dict.keys()))[1])
