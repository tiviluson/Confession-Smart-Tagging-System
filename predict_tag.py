import gensim, re
import numpy as np
import pandas as pd
import pickle
from os import listdir

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer, tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences

import sys
import os

from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Embedding

from underthesea import word_tokenize

sep = os.sep # directory separator
data_folder = "data" # folder that contains data and model
data_file = "Data.csv"
model_version = "final"

enable_train_new_model = True # Set this to False if you want to reuse an existing model with model_version

PAD_LEN = 500 # The maximum length of a sentence

# Dictionary to scale the dataset for a more balanced dataset
freq = dict({("#tìmngườiyêu", 3), ("#lcd", 3), ("#gópý", 18), ("#bócphốt", 10), ("#hỏiđáp", 2), ("#tìmbạn", 2), ("#tâmsự", 1), ("#chiasẻ", 1)})

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
    # Split sentences according to ; * \n . ? !
    text = re.split('; |\*|\n|\.|\?|\!', sentences)

    # Remove the " \ /
    text = [re.sub(r'|,|"|\\|\/', '', sentence) for sentence in text]

    # VNmese compound noun
    text = [word_tokenize(sentence, format="text") for sentence in text]

    # lowercase everything and remove all unnecessary spaces
    text = [sentence.lower().strip().split() for sentence in text if sentence != '']
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
        sentences = preProcess(sample)
        tag = df.loc[df.content == sample, 'tag'].values[0]
        label = [preProcessTag(tag) for _ in sentences]
        # [tag tag tag tag tag]
        for i in range(freq[preProcessTag(tag)]):
            texts = texts + sentences
            labels = labels + label
    return texts, labels

# Train an entirely new model and save it
def trainData():
    dataframe = loadDataFromCSV()
    texts, labels = loadData(dataframe)
    tokenizer, word_index = txtTokenizer(texts)

    # Save the tokenizer
    json_tokenizer = tokenizer.to_json()
    json_file = open(data_folder + sep + "tokenizer_" + model_version + ".json", "w")
    json_file.write(json_tokenizer)
    json_file.close()

    # Put the tokens in a matrix
    X = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(X, maxlen=PAD_LEN)

    # Prepare the labels
    Y = pd.get_dummies(labels)

    # Save the new model
    file = open(data_folder + sep + "data_" + model_version + ".pkl", 'wb')
    pickle.dump([X,Y, texts],file)
    file.close()

    # Train Word2Vec model on our data
    word_model = gensim.models.Word2Vec(texts, vector_size=300, min_count=1, epochs=10)
    word_model.save(data_folder + sep + "word_model_" + model_version + ".save")

    embedding_matrix = np.zeros((len(word_model.wv) + 1, 300))
    for i, vec in enumerate(word_model.wv.vectors):
        embedding_matrix[i] = vec

    # Split train an test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, shuffle=True)

    model = Sequential()
    model.add(Embedding(len(word_model.wv)+1,300,input_length=X.shape[1],weights=[embedding_matrix],trainable=False))
    model.add(LSTM(300,return_sequences=False))
    model.add(Dense(Y.shape[1],activation="softmax"))
    model.summary()
    model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=['acc'])

    batch = 64
    epochs = 10
    model.fit(X_train,Y_train,batch,epochs)
    model.save(data_folder + sep + "predict_model_" + model_version + ".save")
    model.evaluate(X_test, Y_test)

    return tokenizer, model, Y_train.columns

# Reload all required data for reusing a model
def reloadData():
    file = open(data_folder + sep + "tokenizer_" + model_version + ".json")
    tokenizer = tokenizer_from_json(file.read())
    file = open(data_folder + sep + "data_" + model_version + ".pkl", 'rb')
    X, Y, texts = pickle.load(file)
    file.close()
    # Split train an test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, shuffle=True)
    # word_model = gensim.models.Word2Vec.load(data_folder + sep + "word_model_" + model_version + ".save")
    model = load_model(data_folder + sep + "predict_model_" + model_version + ".save")
    return tokenizer, model, Y_train.columns

def predictTag(tokenizer, model, labels):
    # Test model
    file = open(data_folder + sep + "My_data.txt", "r", encoding="utf8")
    input_string = file.read()
    X_dev = tokenizer.texts_to_sequences(preProcess(input_string))
    X_dev = pad_sequences(X_dev, maxlen=PAD_LEN)
    print("Predicting...")
    result_prediction_dict = dict()
    prediction_cus = model.predict(X_dev, verbose=1)
    print(tokenizer.sequences_to_texts(X_dev))
    for i in range(len(prediction_cus)):
        result_tag = labels[np.argmax(prediction_cus[i])]
        result_prediction_dict[result_tag] = result_prediction_dict.get(result_tag, 0) + 1
    print(result_prediction_dict)
    print(max(zip(result_prediction_dict.values(), result_prediction_dict.keys()))[1])

if __name__ == '__main__':
    tokenizer = None
    model = None
    labels = None
    if enable_train_new_model:
        tokenizer, model, labels = trainData()
    else:
        tokenizer, model, labels = reloadData()
    predictTag(tokenizer, model, labels)
