import string
import nltk
nltk.download('punkt')
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
BATCH_SIZE=1024
EPOCHS = 100
clip = 1 
filename ='data/librispeech_100.txt'
output_dir="out/sgns_B"
vecs_filename = "words.vec"
model_filename="model.pt"

#scale
s=1

#model configuration
char_embed_size=50 
gru_size=50*s
gru_layers=1*s
output_dir="out/sgns_B_s"+str(s)
negatives = 10
lin_layers=1*s
adam_eps=1e-8

if s>2: #for numerical stability in larger models
   adam_eps=1e-4



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

"""Stage1 - Char Model """

input_dim = len(char2id)
print('number of character %i' %(input_dim))
encoder_input_dim = input_dim  
decoder_hidden_dim = char_embed_size*2
encoder_hidden_dim = char_embed_size
decoder_output_dim = input_dim

data_all=[]
for i in range(1, len (vocab)):
  data_all.append(word2chars[idx2word[i]])

train_dataset = datasets.CharsDataset(data_all)
def pad_collate(batch):
  xx = batch
  xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
  return xx_pad

data_loader = DataLoader(dataset=train_dataset, batch_size=62, shuffle=True, collate_fn=pad_collate)
enc = models.Encoder(encoder_input_dim, encoder_hidden_dim, hid_dim=gru_size, n_layers=gru_layers, device=device)
dec = models.Decoder(encoder_input_dim, gru_size*2, n_layers=gru_layers)
model = models.Seq2Seq(enc,dec,device).to(device)
optimizer = optim.Adam(model.parameters())
cc_loss = nn.CrossEntropyLoss(ignore_index=0)
print(model)

model.train()
losses = []
best_loss = math.inf
for epo in range(EPOCHS):
  epoch_loss = 0
  model.train()
  for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
      src = batch.to(device)      
      optimizer.zero_grad()
      output,_ = model(src)
      output_dim = output.shape[-1]
      output = output[0:].view(-1, output_dim)
      src = src[0:].view(-1)
      loss = cc_loss(output,src)     
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), clip)   
      optimizer.step()
      epoch_loss += loss.item()

  epoch_loss /= len(data_loader)
  losses.append(epoch_loss)
  print("Epoch:{}, train loss:{} ".format(epo + 1,epoch_loss))

model.eval()
data_all=[]
words=[]
for i in range(1, len(vocab)):
  word=idx2word[i]
  words.append(word)
  chars=[]
  for char in list(word):
       chars.append(char2id[char])
  data_all.append(torch.LongTensor(chars))

data_loader = DataLoader(dataset=data_all, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate)
char_vectors=[]
for i, batch in tqdm(enumerate(data_loader)):
      src = batch.to(device)            
      _,h = model(src)
      res=h.detach().cpu().unsqueeze(0).numpy().tolist()
      for emb in res:
          char_vectors.append(emb)
char_embeddings = np.concatenate(char_vectors, 0)
  
"""Stage 2 - SGNS Model """
all_words=torch.LongTensor(range(1, len(vocab)))
output_data_loader=data_loader = DataLoader(dataset=all_words, batch_size=BATCH_SIZE, shuffle=False)
def output_word_vecs(w_encoder, out_filename=vecs_filename):
  w_encoder.eval()
  final_char_vectors=[]
  for i, batch in tqdm(enumerate(output_data_loader)):
        src = batch.to(device)         
        h = w_encoder(src)
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

print("\n\nBuilding SGNS model with fixed orthographic embeddings from char encoder")
_w=[]
_w.append(np.zeros(gru_size*2).tolist())
for i in range(1, len(idx2word)):
  _w.append(char_embeddings[i-1])
emb_weights=torch.FloatTensor(_w)

input_dim=len(vocab)
sgns_w_encoder=models.WordEmbedding(input_dim, gru_size*2, pretrained=emb_weights, n_layers=lin_layers, device=device).to(device)
sgns_c_encoder=models.WordEmbedding(input_dim, gru_size*2, pretrained=emb_weights, n_layers=lin_layers, device=device).to(device)
params = list(sgns_w_encoder.parameters()) + list(sgns_c_encoder.parameters())
optimizer = optim.Adam(params, eps = adam_eps)


train_dataset = datasets.WordsCBOWDataset(s_sentences, word2idx, window=WINDOW)

print(sgns_w_encoder)
print(sgns_c_encoder)

data_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

losses = []
best_loss = math.inf
sgns_w_encoder.train()
sgns_c_encoder.train()

#sampling from the whole set:
all_word_ids=[]
for s in s_sentences:
  all_word_ids.extend([word2idx[w] for w in s])

all_word_ids=torch.LongTensor(all_word_ids)

for epo in range(EPOCHS):
  epoch_loss = 0
  for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
      tgs = batch[:,0].to(device)
      pos=batch[:,1:].to(device) 

      #NO SUBSAMPLING
      samples = FT(pos.shape[0]*WINDOW * negatives).uniform_(0, len(all_word_ids) - 1).long()
      neg = all_word_ids[samples]
      neg= neg.view(pos.shape[0], WINDOW * negatives).to(device)

      tgs=sgns_w_encoder(tgs).unsqueeze(2)
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


