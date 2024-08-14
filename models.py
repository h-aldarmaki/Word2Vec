import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor as FT

"""#Char-Based Encoder"""

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_size=20, hid_dim=50, n_layers=1, device='cuda'):
        super().__init__()
        self.device = torch.device(device)
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.emb_size=emb_size
        self.embed = nn.Embedding(input_dim, emb_size)
        self.rnn = nn.GRU(emb_size, hid_dim, n_layers, bidirectional=True, batch_first = True)

    def forward(self, src):
        emb=self.embed(src)    
        outputs, hidden = self.rnn(emb) 
        return hidden[self.n_layers*2-2:] 

class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim=100, n_layers=1):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = hid_dim  #To add attention to the encoder's last hidden state
        self.hid_dim = hid_dim
        self.n_layers = n_layers 
        self.rnn = nn.GRU(self.input_dim, self.hid_dim, n_layers, bidirectional=False, batch_first = True)       
        self.fc_out = nn.Linear(hid_dim, self.output_dim)        
        
    def forward(self, hidden, hidden_enc):     
        #print(hidden_enc.shape)
        hidden_enc = hidden_enc.unsqueeze(1)    
        #print(hidden_enc.shape)
        #print(hidden.shape)
        output, hidden = self.rnn(hidden_enc, hidden)       
        prediction = self.fc_out(output.squeeze(1))        
        return prediction, hidden#.squeeze(0)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()    
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, teacher_forcing_ratio = 0):
        batch_size = src.shape[0]
        decoder_hidden_size = self.decoder.hid_dim
        
        hidden_enc = self.encoder(src)
        hidden_enc = torch.cat((hidden_enc[0],hidden_enc[1]), dim=1)#.unsqueeze(1)
        hidden=torch.zeros(self.decoder.n_layers, batch_size, decoder_hidden_size).to(self.device)
        trg_len=src.shape[1]
        outputs = torch.zeros(batch_size, trg_len, self.decoder.output_dim).to(self.device)
        for t in range(0, trg_len):    
            output, hidden = self.decoder(hidden, hidden_enc)
            outputs[:,t,:] = output.squeeze(1)    
        return outputs, hidden_enc

class WordEmbedding(nn.Module):
    def __init__(self, input_dim, emb_size, pretrained=None, n_layers=0, device='cuda'):
        super().__init__()
        self.n_layers=n_layers
        self.emb_size=emb_size
        self.device=device
        self.embed = nn.Embedding(input_dim, emb_size)
        self.n_layers=n_layers #number of feedforward layers
        self.f = nn.ModuleList()
        if pretrained is None:
          self.embed.weight =nn.Parameter(torch.cat([torch.zeros(1, self.emb_size), FT(input_dim - 1, self.emb_size).uniform_(-0.5 / self.emb_size, 0.5 / self.emb_size)]))
          self.embed.weight.requires_grad = True
        else:
          self.embed =  nn.Embedding.from_pretrained(pretrained)
          self.embed.weight.requires_grad = False
        for i in range(n_layers):
           self.f.append(nn.Linear(self.emb_size, self.emb_size).to(device))

    def forward(self, src):
        emb=self.embed(src)
        for i in range(self.n_layers):
          emb=self.f[i](emb)
        return emb
   
class WordEncoder(nn.Module):
    def __init__(self, input_dim, context=3, emb_size=100, hid_dim=200, emb_layers=0, n_layers=0, device='cuda', pretrained=None):
        super().__init__()
        self.device = torch.device(device)
        self.hid_dim = hid_dim
        self.emb_size=emb_size
        self.context=context
        self.input_dim=input_dim
        self.n_layers=n_layers

        self.embed = WordEmbedding(input_dim, emb_size, pretrained=pretrained, n_layers=emb_layers,device=device)
        
        self.f =nn.ModuleList()
        _h_size=self.emb_size*context
        for i in range(n_layers):
          self.f.append(nn.Linear(_h_size, self.hid_dim).to(device))
          _h_size=self.hid_dim

        self.f_output=nn.Linear(_h_size, self.input_dim)
      
    def forward(self, src):
        src=torch.empty_like(src).bernoulli_(0.7).to(self.device) * src
        emb=self.embed(src)
        _h=torch.flatten(emb, start_dim=1)
        for i in range(self.n_layers):   
          _h=self.f[i](_h)

        out=self.f_output(_h)
        return out


