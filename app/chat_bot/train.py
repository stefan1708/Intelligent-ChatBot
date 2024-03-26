from model import ChatBotNet
from create_data import create_training_data, ChatDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import torch

def train(model, epochs, dl, criterion, optimizer, save_state_dict, device):
    model.to(device)
    print('Started training ...')
    for epoch in range(epochs):
        for words, labels in dl:
            words = words.to(device)
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)

            outputs = model(words)
            loss = criterion(outputs, labels)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch}/{epochs}]  |  loss: {loss}')
    torch.save(model.state_dict(), save_state_dict)
    print('Finished training')
    
if __name__ == '__main__':

    args = {
        'input_size': 188,
        'num_classes': 20,
        'epochs': 500,
        'criterion': CrossEntropyLoss(),
        'save_state_dict': 'C:/Users/stefa/OneDrive/Desktop/ChatBot/app/chat_bot/chatbot_model.pt',
        'device': 'cpu',
        'intents_json': 'C:/Users/stefa/OneDrive/Desktop/ChatBot/app/chat_bot/intents.json'
    }
    
    x_train, y_train, _, __ = create_training_data(args.get('intents_json'))
    ds = ChatDataset(x_train, y_train)
    dl = DataLoader(ds, batch_size=len(x_train))

    model = ChatBotNet(input_size=args.get('input_size'), num_classes=args.get('num_classes'))
    optimizer = Adam(model.parameters())

    train(model,
        args.get('epochs'), 
        dl, args.get('criterion'), 
        optimizer, 
        args.get('save_state_dict'), 
        args.get('device'))