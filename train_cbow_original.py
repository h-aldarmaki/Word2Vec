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
output_dir="out/cbow_original_clean100"
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

train_dataset = datasets.WordsCBOWDataset(s_sentences, word2idx, window=WINDOW)

input_dim=len(vocab)
cbow_encoder=models.WordEncoder(input_dim, WINDOW*2, device=device).to(device)
optimizer = optim.Adam(cbow_encoder.parameters())
print(cbow_encoder)

data_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

#Training Loop: CBOW model (original)

losses = []
best_loss = math.inf
cc_loss=nn.CrossEntropyLoss()
cbow_encoder.train()
for epo in range(EPOCHS):
  epoch_loss = 0
  for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
      tgs = batch[:,0].to(device)
      pos=batch[:,1:].to(device) 
      out=cbow_encoder(pos)
      optimizer.zero_grad()
      output_dim = out.shape[-1]
      loss=cc_loss(out, tgs)
      loss.backward()           
      optimizer.step()
      epoch_loss += loss.item()

  epoch_loss /= len(data_loader)
  losses.append(epoch_loss)
  print("Epoch:{}, train loss:{} ".format(epo + 1,epoch_loss))


#store checkpoint
torch.save({'epoch': epo,
            'model_state_dict': cbow_encoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
            }, os.path.join(output_dir, model_filename))

cbow_encoder.eval()

#encode all words
all_words=torch.LongTensor(range(0, len(word2idx.keys())))
data_loader = DataLoader(dataset=all_words, batch_size=BATCH_SIZE, shuffle=False)
word_vecs=[]
for i, batch in tqdm(enumerate(data_loader)):
      h = cbow_encoder.embed(batch.to(device))
      res=h.detach().cpu().unsqueeze(0).numpy().tolist()
      for emb in res:
          word_vecs.append(emb)

word_vecs = np.concatenate(word_vecs, 0)
word_vecs = torch.from_numpy(word_vecs).float()
words =[k for k in word2idx.keys()]
utils.export_embeddings(words, word_vecs, os.path.join(output_dir,vecs_filename))

