import torch
from tqdm import tqdm

class CharsDataset(torch.utils.data.Dataset):
  def __init__(self, words):
        'Initialization'
        self.words = words

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.words)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = self.words[index]
        return X

#return skipgram positive examples for a given line
def construct_skipgram_examples(line, word2idx, window=5, UNK='[UNK]'):
  res = []
  words=line
  for i in range(len(words)):
    left = line[max(i - window, 0): i]
    right = line[i + 1: i + 1 + window]
    temp=[word2idx[words[i]]]
    pos=[UNK for _ in range(window - len(left))] + left + right + [UNK for _ in range(window - len(right))]
    value = list(map(word2idx.get, pos)) 
    temp.extend(value)
    res.append(torch.LongTensor(temp))
                               
  return res

def construct_LM_examples(line, window=4):
  res = []
  words=line
  for i in range(0, len(words)-window):
    context=[]
    for j in range(window):
      context.append(word2idx[words[i+j]])
    res.append(torch.LongTensor(context))
  return res

def process_line(line, word2idx, UNK):
  for i in range(len(line)):
    if line[i] not in word2idx.keys():
      line[i]=UNK
  return line

class WordsCBOWDataset(torch.utils.data.Dataset):
  def __init__(self, lines, vocab, window=5, UNK='[UNK]'):
        'Initialization'
        self.words = []
        self.window = window
        for line in tqdm(lines, total=len(lines)):
          line=process_line(line, vocab, UNK)
          self.words.extend(construct_skipgram_examples(line, vocab, window, UNK))

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.words)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = self.words[index]
        return X


#sumsampling frequent words for skipgram model
def subSample(sentences, unigrams):
  words = [w for sent in sentences for w in sent]
  res = []
  for sent in sentences:
    new_s=[]
    for word in sent:
      frac = unigrams.counts[word]/len(words)
      prob = (np.sqrt(frac/0.001) + 1) * (0.001/frac)   
      if np.random.random() < prob:
         new_s.append(word)
    if len(new_s)>0:
         res.append(new_s)
  return res