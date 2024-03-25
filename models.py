import torch
import torch.nn as nn

# The Encoder network

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(101, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 4)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)

    def forward(self, att):


        att = self.relu(self.bn1(self.fc1(att)))
        att = self.dropout(att)

        att = self.relu(self.bn2(self.fc2(att)))
        att = self.dropout(att)

        att = self.relu(self.fc3(att))

        return att


# The Decoder network
        
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
 
        self.fc1 = nn.Linear(4, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 101)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.LeakyReLU(0.2)
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)

    def forward(self, att_4_mse):

        att = self.relu(self.bn1(self.fc1(att_4_mse)))
        att = self.dropout(att)

        att = self.relu(self.bn2(self.fc2(att)))
        att = self.dropout(att)

        att = self.fc3(att)

        return att  
