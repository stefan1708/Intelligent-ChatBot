import torch.nn as nn
from torchsummary import summary

class ChatBotNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU()
        )
        self.l2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.l3 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        return out
    
if __name__ == '__main__':

    model = ChatBotNet(input_size=63, num_classes=7)
    model.to('cuda')
    summary(model, input_size=(1, 63))