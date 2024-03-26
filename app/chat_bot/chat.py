from model import ChatBotNet
from create_data import tokenize, bag_of_words, create_training_data
import torch
import json
import random

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

bot_name = 'ChatBot'
print("Hai sa vorbim! scrie 'pa' pentru a incheia discutia!")

while True:
    sentence = input('You: ')
    if sentence.lower() == 'pa':
        break
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    
    output = model(X)
    _, pred = torch.max(output, dim=1)
    tag = tags[pred.item()]
    
    probs = torch.softmax(output, dim=1)
    prob = probs[0][pred.item()]
    
    if prob.item() > 0.3:
        for intent in intents['intents']:
            if tag == intent['tag']:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
        
    else:
        print(f"{bot_name}: Nu inteleg intrebarea!")
        
print(f'{bot_name}: Pe data viitoare!')