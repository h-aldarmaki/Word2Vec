import string
import nltk
from nltk.tokenize import word_tokenize
import utils
import datasets
import models
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils import data
import torch.nn.functional as F
from tqdm  import tqdm
from torch import FloatTensor as FT
import math
import os
from torch.nn.utils.rnn import pad_sequence
from gensim.models import KeyedVectors, Word2Vec
from scipy import stats


#settings
UNK='[UNK]'
PAD='[PAD]'
max_vocab=20000 
WINDOW=5
negatives = 20
device = "cuda"
BATCH_SIZE=512
EPOCHS = 100
clip = 1 

#scale
s=1

#model config
char_embed_size=50 
gru_size=50*s
gru_layers=1*s
output_dir="out/sgns_C_s"+str(s)
negatives = 10
lin_layers=1*s
adam_eps = 1e-8

if s>2:
   adam_eps = 1e-6



filename ='data/librispeech_100.txt'
vecs_filename = "words.vec"
model_filename="model.pt"

print('\nProcessing input file:', filename)
#process dataset
file = open(filename, 'rt')
text = file.readlines()
file.close()

n_s=0
n_w=0
s_sentences=[]
word_idx=0
for sent in text:
  new_sen=[]
  tokens=word_tokenize(sent)
  for word in tokens:#sent.split():
   # if word not in string.punctuation:
    word = word.lower()
    new_sen.append(word.lower())
    n_w+=1
  s_sentences.append(new_sen)
  n_s+=1

print("Number of sentences:", n_s)
print("Number of words:", n_w)

vocab, word2idx, idx2word = utils.build_vocab(s_sentences, UNK=UNK, max_vocab=max_vocab)

word2chars, char2id= utils.build_char_vocab(idx2word, PAD=PAD)

"""Create Data """

#Below, load examples. Covert positive examples to sequences of characters. Convert targets to list
train_dataset = datasets.WordsCBOWDataset(s_sentences, word2idx, WINDOW)
data_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False)

print('\n\nConstructing Character-Based Corpus - Sequences of Context Words, each is a padded sequence of chars')
_features = []
_targets =[]

max_word_len=0
for word in vocab:
  if len(word)>max_word_len:
    max_word_len=len(word)

word2chars[UNK]=torch.LongTensor(np.zeros(max_word_len))

for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
    words=batch[:][0].tolist()
    _temp=[]
    for _word in words:
      word=idx2word[_word]
      if word not in word2chars:
        chars=[]
        for char in list(word):
          if char in char2id:
            chars.append(char2id[char])
          else:
            chars.append(char2id[PAD])

        chars.extend(0 for _ in range(max_word_len-len(chars)))
        _temp.append(chars[:max_word_len])
      else:
        chars=word2chars[word].tolist()
        chars.extend(0 for _ in range(max_word_len-len(chars)))
        _temp.append(chars[:max_word_len])
    _features.append(torch.LongTensor(_temp[1:]))
    _targets.append(torch.LongTensor(_temp[0]))


#updated classes

device='cuda'
class CharEncoder(nn.Module):
    def __init__(self, input_dim, emb_size=50, hid_dim=50, n_layers=2, gru_layers=1, device='cuda', pretrained=None):
        super().__init__()
        self.device = torch.device(device)
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.gru_layers=gru_layers
        self.emb_size=emb_size
        self.embed = nn.Embedding(input_dim, emb_size)
        if pretrained is not None:
          #load from pretrained
          self.embed=nn.Embedding.from_pretrained(pretrained)
          self.embed.weight.requires_grad = False

        self.rnn = nn.GRU(emb_size, hid_dim, num_layers=gru_layers, bidirectional=True, batch_first = True)
        self.f=nn.ModuleList()
        for i in range(self.n_layers):
          self.f.append(nn.Linear(hid_dim*2, hid_dim*2).to(device))

    def forward(self, src):
        emb=self.embed(src)    
        emb=emb.view(-1, emb.shape[2], emb.shape[3])
        #print(emb.shape)
        outputs, hidden = self.rnn(emb) 
        hidden=hidden[self.gru_layers*2-2:] 
        hidden=hidden.view(hidden.shape[0], src.shape[0], -1, self.hid_dim)
        hidden_enc = torch.cat((hidden[0],hidden[1]), dim=2)
        for i in range(self.n_layers):
          hidden_enc = self.f[i](hidden_enc)
        return hidden_enc



def pad_collate(batch):
  feats = [batch[i][0] for i in range(len(batch))] 
  targs = [batch[i][1].tolist() for i in range(len(batch))] 
  #print(targs)
  padded_feats = pad_sequence(feats, batch_first=True, padding_value=0)  #then set ignore index = 0
  return padded_feats, torch.LongTensor(targs)

class AudioDataset(Dataset):
  def __init__(self, features, targets):
        'Initialization'
        self.targets= targets #these should be individual word ids
        self.features = features #each line should be a 2D tensor, where the rows are words, and columns are char ids

  def __len__(self):
        return len(self.features)

  def __getitem__(self, index):
        'Generates one sample of data'
        return self.features[index], self.targets[index]



audio_train_data=AudioDataset(_features, _targets)

data_loader = DataLoader(dataset=audio_train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate)

input_dim = len(char2id)
print('number of character %i' %(input_dim))

encoder_input_dim = input_dim  
encoder_hidden_dim = 50

print("\n\nBuilding SGNS model with integrated char encoder")

sgns_w_encoder = CharEncoder(encoder_input_dim,  emb_size=char_embed_size, hid_dim=gru_size, n_layers=2, gru_layers=gru_layers, device=device).to(device)
sgns_c_encoder = CharEncoder(encoder_input_dim,  emb_size=char_embed_size, hid_dim=gru_size, n_layers=2, gru_layers=gru_layers, device=device).to(device)

print(sgns_w_encoder)
print(sgns_c_encoder)

encoder_total_params = sum(p.numel() for p in sgns_w_encoder.parameters())
print("Word encoder parameters: ", encoder_total_params)

params = list(sgns_w_encoder.parameters()) + list(sgns_c_encoder.parameters())
optimizer = optim.Adam(params, eps = adam_eps)

cc_loss=nn.CrossEntropyLoss()
sgns_w_encoder.train()
sgns_c_encoder.train() 

#Negative samples pool
all_word_chars=[]
for k, v in word2chars.items():
  temp=v.tolist()
  temp.extend(0 for _ in range(max_word_len-len(v.tolist())))
  all_word_chars.append(temp[:max_word_len])
all_word_chars=torch.LongTensor(all_word_chars)


#create dataloader for saving word embeddings:
def char_pad_collate(batch):
  xx = batch
  xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
  return xx_pad

"""Create data for embedding """
data_all=[]
words=[]
for i in range(1, len(vocab)):
  word=idx2word[i]
  words.append(word)
  chars=[]
  for char in list(word):
       chars.append(char2id[char])
  data_all.append(torch.LongTensor(chars))

output_data_loader = DataLoader(dataset=data_all, batch_size=BATCH_SIZE, shuffle=False, collate_fn=char_pad_collate)

def output_word_vecs(w_encoder, out_filename=vecs_filename):
  w_encoder.eval()
  final_char_vectors=[]
  for i, batch in tqdm(enumerate(output_data_loader)):
        src = batch.to(device).unsqueeze(1)          
        h = w_encoder(src).squeeze(1)
        res=h.detach().cpu().unsqueeze(0).numpy().tolist()
        for emb in res:
            final_char_vectors.append(emb)


  word_vecs = np.concatenate(final_char_vectors, 0)
  word_vecs = torch.from_numpy(word_vecs).float()
  utils.export_embeddings(words, word_vecs, os.path.join(output_dir,out_filename))
  w_encoder.train()
  return word_vecs

#models for eval
#load c model
c_model = KeyedVectors.load_word2vec_format('char_embeddings.vec')

#train gensim target model
print("training target model ... ")
t_model=Word2Vec(s_sentences, epochs=100)
#create test set
print("creating test set ... ")
test_set=utils.create_test_set(s_sentences, t_model.wv, c_model)




"""#Train model"""

sgns_w_encoder.train()
sgns_c_encoder.train()
losses = []
best_loss = math.inf
for epo in range(EPOCHS):
  epoch_loss = 0
  for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
      tgs = batch[1].to(device)
      pos=batch[0].to(device) 

      samples = FT(pos.shape[0]*WINDOW * negatives).uniform_(0, len(all_word_chars) - 1).long()
      neg = all_word_chars[samples]
      neg= neg.view(pos.shape[0], WINDOW * negatives, -1).to(device)

      tgs=sgns_w_encoder(tgs.unsqueeze(1)).squeeze(1).unsqueeze(2)
      pos=sgns_c_encoder(pos)
      neg=sgns_c_encoder(neg).neg()
      optimizer.zero_grad()

      oloss = torch.bmm(pos, tgs).squeeze().sigmoid().log().mean(1)
      nloss = torch.bmm(neg, tgs).squeeze().sigmoid().log().view(-1, WINDOW, negatives).sum(2).mean(1)
      loss= -(oloss + nloss).mean()

      loss.backward()           
      optimizer.step()
      epoch_loss += loss.item()

  epoch_loss /= len(data_loader)
  losses.append(epoch_loss)
  print("Epoch:{}, train loss:{} ".format(epo + 1,epoch_loss))
  #check if epo is a multiple of 10
  #eval_vectors()
  if epoch_loss<best_loss:#epo%10 == 0:
    best_loss=epoch_loss
    #save model
    print("saving model ... ")
    torch.save({'epoch': epo,
            'model_state_dict': sgns_w_encoder.state_dict(),
            'model_state_dict_c': sgns_c_encoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
            }, os.path.join(output_dir, model_filename))
    #output embeddings
    word_vecs=output_word_vecs(sgns_w_encoder)

  word_vecs=output_word_vecs(sgns_w_encoder, 'temp.vec')
  s_model = KeyedVectors.load_word2vec_format(os.path.join(output_dir,'temp.vec'))
  #eval source embeddings
  X=utils.calc_test_sim(s_model, t_model.wv, test_set)
  Y=utils.calc_test_sim(t_model.wv, s_model,  test_set)
  Z=utils.cal_test_edit(t_model.wv, s_model, test_set)
  correlation, p_value = stats.pearsonr(X, Y)
  print("Pearson correlation w. target model sim sim:", correlation)
  correlation, p_value = stats.pearsonr(X, Z)
  print("Pearson correlation w. edit distance:", correlation)





