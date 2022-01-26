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

import keras

from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Embedding

from underthesea import word_tokenize

import matplotlib.pyplot as plt
import tensorflow as tf

from googletrans import Translator, constants
import math

sep = os.sep # directory separator
data_folder = "data" # folder that contains data and model
data_file = "Data_final.csv"
model_version = "test_switch_oversampling_spliting_5"

enable_train_new_model = False # Set this to False if you want to reuse an existing model with model_version

PAD_LEN = 500 # The maximum length of a sentence

def loadDataFromCSV():
    df = pd.read_csv(data_folder + sep + data_file)
    df['tag'] = df['tag'].fillna("#LCD")
    return df

# Create a tokenizer to use later on
def txtTokenizer(texts):
    # texts: A list of sentences
    tokenizer = Tokenizer()
    # Fit the tokenizer on our text
    tokenizer.fit_on_texts(texts)

    # Get all words that the tokenizer knows
    word_index = tokenizer.word_index
    return tokenizer, word_index

# Remove trash symbols and spaces + Lower the case of all data
def preProcess(sentences):
    # Split sentences according to ; * \n . ? !
    text = re.split('; |\*|\n|\.|\?|\!', sentences)

    # Remove the " \ / ,
    text = [re.sub(r'|,|"|\\|\/', '', sentence) for sentence in text]

    # VNmese compound words
    text = [word_tokenize(sentence, format="text") for sentence in text]

    # lowercase everything and remove all unnecessary spaces
    text = [sentence.lower().strip().split() for sentence in text if sentence != '']

    return text

# Preprocess all confessions
def preProcessTextInDataFrame(df):
    texts_labels_df = pd.DataFrame(columns = df.columns)
    for idx, row in df.iterrows():
        confession = row[0]
        sentences = preProcess(confession)
        for sentence in sentences:
            temp_row = row
            temp_row[0] = sentence
            texts_labels_df = texts_labels_df.append(temp_row, ignore_index=True)
    return texts_labels_df

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
    texts_labels_df = pd.DataFrame(df[["content", "tag"]])
    texts_labels_df.columns = ["Texts", "Labels"]
    for index, data in texts_labels_df.iterrows():
        texts_labels_df.loc[index, "Labels"] = preProcessTag(data[1])
    return texts_labels_df # [cf1, tag1]
                           # [cf2, tag2]
                           # [cf3, tag3]

# Create a scaling dictionary for the 8 tags to multiply later in oversampling
def createScaleDict(df):
    tag_frequency = df.iloc[:,1:].apply(pd.value_counts).iloc[1]
    max_frequency = tag_frequency.max()
    scale_dict = {}
    for index, frequency in tag_frequency.iteritems():
        scale_dict[index] = math.floor(max_frequency / frequency)
    return scale_dict

def backTranslate(translator, txt, lang):
    translation = translator.translate(txt, src="vi", dest=lang)
    if translation.text == "" or translation.text == None or len(translation.text) > 5000:
        return txt
    translation_to_vi = translator.translate(translation.text, src=lang, dest='vi')
    return translation_to_vi.text

def overSamplingData(df, scale_dict):
    new_df = df
    translator = Translator()
    for ind in df.index:
        row = df.loc[ind]
        # Backtranslation
        scale = scale_dict[row[row == 1].index[0]]
        for i in range(scale - 1):
            temp_row = row
            random_language = np.random.choice(list(constants.LANGUAGES.keys()))
            temp_row[0] = temp_row[0].strip()
            if temp_row[0] != "" and temp_row[0] != None and len(temp_row[0]) < 5000 :
                temp_row[0] = backTranslate(translator, temp_row[0], random_language)
                new_df = new_df.append(temp_row, ignore_index=True)
    return new_df

# Train an entirely new model and save it
def trainData():
    df = loadDataFromCSV()
    texts_labels_df = loadData(df)

    # Get dummies
    print("Getting dummies for the tags...")
    texts_labels_dummy_df = pd.get_dummies(data = texts_labels_df, columns = ["Labels"], prefix = "", prefix_sep = "")

    # Split the train and test sets as well as oversampling the data in train set
    print("Splitting the train and test sets...")
    train_df = texts_labels_dummy_df.sample(frac = 0.9, random_state = 69)
    test_df = texts_labels_dummy_df.drop(train_df.index)
    print(train_df)

    # Do oversampling in the train set
    print("Oversampling training data...")
    train_df = overSamplingData(train_df, createScaleDict(train_df))

    # Preprocess the texts in all 3 sets
    print("Preprocessing data...")
    texts_labels_dummy_df = preProcessTextInDataFrame(texts_labels_dummy_df)
    train_df = preProcessTextInDataFrame(train_df)
    test_df = preProcessTextInDataFrame(test_df)
    print(train_df)

    # Create and save the tokenizer
    print("Creating tokenizer...")
    tokenizer, word_index = txtTokenizer(texts_labels_df["Texts"].tolist())
    print("Saving tokenizer...")
    json_tokenizer = tokenizer.to_json()
    json_file = open(data_folder + sep + "tokenizer_" + model_version + ".json", "w")
    json_file.write(json_tokenizer)
    json_file.close()

    # Put the tokens in a matrix
    print("Creating X...")
    X = tokenizer.texts_to_sequences(texts_labels_dummy_df["Texts"].tolist())
    X = pad_sequences(X, maxlen=PAD_LEN)
    X = np.asarray(X).astype(np.int)

    # Prepare the labels
    print("Creating Y...")
    Y = texts_labels_dummy_df.loc[:, texts_labels_dummy_df.columns != "Texts"]

    # Do the same thing as above but for train and test sets
    print("Creating X_train...")
    X_train = tokenizer.texts_to_sequences(train_df["Texts"].tolist())
    X_train = pad_sequences(X_train, maxlen=PAD_LEN)
    X_train = np.asarray(X_train).astype(np.int)
    print("Creating Y_train...")
    Y_train = train_df.loc[:, train_df.columns != "Texts"]
    Y_train = Y_train.apply(pd.to_numeric, errors = "coerce")
    print("Creating X_test...")
    X_test = tokenizer.texts_to_sequences(test_df["Texts"].tolist())
    X_test = pad_sequences(X_test, maxlen=PAD_LEN)
    X_test = np.asarray(X_test).astype(np.int)
    print("Creating Y_test...")
    Y_test = test_df.loc[:, test_df.columns != "Texts"]
    Y_test = Y_test.apply(pd.to_numeric, errors = "coerce")

    # Save the new model
    print("Saving the pickle...")
    file = open(data_folder + sep + "data_" + model_version + ".pkl", 'wb')
    pickle.dump([X, Y, X_train, Y_train, X_test, Y_test], file)
    file.close()

    # Train Word2Vec model on our data
    print("Training Word2Vec...")
    word_model = gensim.models.Word2Vec(texts_labels_dummy_df.loc[:, "Texts"].tolist(), vector_size=300, min_count=1, epochs=20)
    word_model.save(data_folder + sep + "word_model_" + model_version + ".save")

    embedding_matrix = np.zeros((len(word_model.wv) + 1, 300))
    for i, vec in enumerate(word_model.wv.vectors):
        embedding_matrix[i] = vec

    # input model: 1 duy nhất thôi :3 sentence
    # output model: 1 cái array có số chiều = số tag
    #         tag1  tag2 ...
    # câu 1   0.2   0.3 ...
    model = Sequential()

    # biến 1 câu thành 1 list of feature vectors
    # "chào em" biến thành mảng 2 chiều (2, 300)
    # 300 là 300 features
    model.add(Embedding(len(word_model.wv)+1,300,input_length=X.shape[1],weights=[embedding_matrix],trainable=False))

    # 300 này là 300 units
    model.add(LSTM(300,return_sequences=False))
    model.add(Dense(Y.shape[1],activation="softmax"))
    model.summary()
    model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=['acc'])

    batch = 64 # mỗi lần train 64 data cùng lúc
    epochs = 20 # train 20 lần
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    history = model.fit(X_train, Y_train, batch, epochs, callbacks=[callback])

    model.save(data_folder + sep + "predict_model_" + model_version + ".save")
    model.evaluate(X_test, Y_test)

    return tokenizer, model, Y_train.columns

# Reload all required data for reusing a model
def reloadData():
    file = open(data_folder + sep + "tokenizer_" + model_version + ".json")
    tokenizer = tokenizer_from_json(file.read())
    file = open(data_folder + sep + "data_" + model_version + ".pkl", 'rb')
    X, Y, X_train, Y_train, X_test, Y_test = pickle.load(file)
    file.close()
    model = load_model(data_folder + sep + "predict_model_" + model_version + ".save")
    model.summary()
    return tokenizer, model, Y.columns

def predictTag(tokenizer, model, labels):
    # Test model
    file = open(data_folder + sep + "My_data.txt", "r", encoding="utf8")
    input_string = file.read()
    X_dev = tokenizer.texts_to_sequences(preProcess(input_string))
    print(X_dev)
    X_dev = pad_sequences(X_dev, maxlen=PAD_LEN)

    intermediate_layer_model = keras.Model(inputs=model.input, outputs=model.get_layer(index=1).output)
    print("-"*20)
    intermediate_layer_model.summary()
    print(intermediate_layer_model(X_dev))
    print("-"*20)


    print("Predicting...")
    result_prediction_dict = dict()
    prediction_cus = model.predict(X_dev, verbose=1)
    print(input_string)
    print(tokenizer.sequences_to_texts(X_dev))
    for i in range(len(prediction_cus)):
        result_tag = labels[np.argmax(prediction_cus[i])]
        result_prediction_dict[result_tag] = result_prediction_dict.get(result_tag, 0) + 1
    print(result_prediction_dict)
    print(max(zip(result_prediction_dict.values(), result_prediction_dict.keys()))[1])

tokenizer = None
model = None
labels = None
if enable_train_new_model:
    tokenizer, model, labels = trainData()
else:
    tokenizer, model, labels = reloadData()
predictTag(tokenizer, model, labels)
