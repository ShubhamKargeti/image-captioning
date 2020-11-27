import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        features = self.bn(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.embed = nn.Embedding(vocab_size, embed_size)
        
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,dropout=0.5,batch_first=True)
        
        self.linear = nn.Linear(hidden_size, vocab_size)
        
        self.softmax = nn.Softmax(dim=2)
    
    def forward(self, features, captions):
        batch_size = features.size(0)
        
        main_caption = captions[:, :-1]
        captions = self.embed(main_caption)
        
        features = features.unsqueeze(1)
        inputs = torch.cat((features, captions), 1)
        lstm_output, _ = self.lstm(inputs, None)#hidden_State - none
        
        outputs = self.linear(lstm_output)
        return outputs

    def sample(self, inputs, states=None, max_len=20, stop_idx=1):
        caption = []
        hidden = None
        for i in range(max_len):
            out, hidden = self.lstm(inputs, hidden)
            output = self.linear(out)

            prediction = torch.argmax(output, dim=2)
            predicted_index = prediction.item()
            caption.append(predicted_index)
            
            
            if predicted_index == stop_idx:
                break
            
            # Get the embeddings for the next cycle.
            inputs = self.embed(prediction)

        return caption