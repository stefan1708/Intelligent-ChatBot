import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader

class ChatDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.n_samples = len(x_data)
        self.x_data = x_data
        self.y_data = y_data
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples


def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word, stemmer=None):
    if stemmer is None:
        stemmer = PorterStemmer()
    return stemmer.stem(word)

def bag_of_words(tokenized_sentence, all_words):
    
    tokenized_sentence = [stem(w) for w  in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    
    for i, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[i] = 1.       
    return bag

def create_training_data(intents_json):
    with open(intents_json, 'r') as f:
        intents = json.load(f) 

    all_words = []
    tags = []
    xy = []
    ignore_words = ['?', '!', '.', ',', '/', ':', ';', '%']

    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']:
            w = tokenize(pattern)
            all_words.extend(w)
            xy.append((w, tag))
        
    all_words = [stem(w) for w in set(all_words) if w not in ignore_words]
    all_words = sorted(all_words)
    tags = sorted(tags)

    X_train = []
    y_train = []

    for pattern, tag in xy:
        bag = bag_of_words(pattern, all_words)
        X_train.append(bag)
    
        label = tags.index(tag)
        y_train.append(label)

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    
    return X_train, y_train, all_words, tags

if __name__ == '__main__':

    intents_json = 'C:/Users/stefa/OneDrive/Desktop/ChatBot/app/chat_bot/intents.json'
    x_train, y_train, _, __ = create_training_data(intents_json=intents_json)
    ds = ChatDataset(x_train, y_train)
    dl = DataLoader(ds, batch_size=ds.n_samples, shuffle=True)
    for words, labels in dl:
        print(len(words[0]))
        
        