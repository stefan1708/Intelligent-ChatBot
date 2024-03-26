from chat_bot.model import ChatBotNet
from chat_bot.create_data import tokenize, bag_of_words, create_training_data
import torch
import json
import random


def make_prediction(sentence=None):
    input_size = 188
    num_classes = 20
    intents_json = 'C:/Users/stefa/OneDrive/Desktop/ChatBot/app/chat_bot/intents.json'
    state_dict = 'C:/Users/stefa/OneDrive/Desktop/ChatBot/app/chat_bot/chatbot_model.pt'
    device = 'cpu'
    with open(intents_json, 'r') as f:
        intents = json.load(f)
    a, b, all_words, tags = create_training_data(intents_json)

    model = ChatBotNet(input_size=input_size, num_classes=num_classes)  
    model.load_state_dict(torch.load(state_dict))
    model.to(device)

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    
    output = model(X)
    _, pred = torch.max(output, dim=1)
    tag = tags[pred.item()]
    
    probs = torch.softmax(output, dim=1)
    prob = probs[0][pred.item()]
    
    if prob.item() > 0.1:
        for intent in intents['intents']:
            if tag == intent['tag']:
                return f"{random.choice(intent['responses'])}"
        
    else:
        return "Nu înțeleg XD"