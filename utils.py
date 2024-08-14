import io
import numpy as np
import torch
from Levenshtein import distance as edit_distance
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
from numpy.linalg import norm
import nltk
from pandas.core.common import flatten
from gensim.models import KeyedVectors
from gensim import models
import pandas as pd
import random
from gensim.models import Word2Vec

#from https://github.com/facebookresearch/MUSE/blob/main/src/utils.py
def read_txt_embeddings(emb_path, dim, max_vocab=100000):
    """
    Reload pretrained embeddings from a text file.
    """
    word2id = {}
    vectors = []
    # load pretrained embeddings
    _emb_dim_file = dim
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        for i, line in enumerate(f):
            if i == 0:
                split = line.split()
                assert len(split) == 2
                assert _emb_dim_file == int(split[1])
            else:
                word, vect = line.rstrip().split(' ', 1)
                word = word.lower()
                vect = np.fromstring(vect, sep=' ')
                if np.linalg.norm(vect) == 0:  # avoid to have null embeddings
                    vect[0] = 0.01
                if word in word2id:
                    print("Word '%s' found twice in %s embedding file"
                                        % (word, 'source' if source else 'target'))
                else:
                    if not vect.shape == (_emb_dim_file,):
                        print("Invalid dimension (%i) for %s word '%s' in line %i."
                                        % (vect.shape[0], 'source' if source else 'target', word, i))
                        continue
                    assert vect.shape == (_emb_dim_file,), i
                    word2id[word] = len(word2id)
                    vectors.append(vect[None])
            if len(word2id) >= max_vocab:
                break

    assert len(word2id) == len(vectors)
    print("Loaded %i pre-trained word embeddings." % len(vectors))

    # compute new vocabulary / embeddings
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.concatenate(vectors, 0)
    embeddings = torch.from_numpy(embeddings).float()
    #assert embeddings.size() == (len(dico), params.emb_dim)
    return embeddings, id2word, word2id

def export_embeddings(words, embeddings, file_name):
    print('Writing embeddings to %s ...' % file_name)
    with io.open(file_name, 'w', encoding='utf-8') as f:
        f.write(u"%i %i\n" % embeddings.size())
        for i in range(len(words)):
            f.write(u"%s %s\n" % (words[i], " ".join('%.5f' % x for x in embeddings[i])))

#calculates pairwise edit distance for the given words
def p_editDist(word_list):
    transformed_strings = np.array(word_list).reshape(-1,1)
    A=pdist(transformed_strings,lambda x,y: edit_distance(x[0],y[0])/max(len(x[0]), len(y[0])))
    return A

#calculate pairwise cosine 
def p_cosine(A):
    scores = pdist(A, metric='cosine')
    return scores


def cosine(A, B):
  return np.dot(A,B)/(norm(A)*norm(B))



#normalize values in X to be in [0-1]
def normalize(X):
  mi=min(X)
  ma=max(X)
  #print(mi)
  return [(x-mi)/(ma-mi) for x in X]


def calc_test_sim(vecs1, vecs2, testset):
  res=[]
  for item in testset:
    s=item[0]
    if s in vecs1.key_to_index and s in vecs2.key_to_index:
       sims = [vecs1.similarity(s, w) for w in item[1:] if w in vecs1.key_to_index and w in vecs2.key_to_index]
       res.extend(sims)
  return res

def calc_test_dist(vecs1, vecs2, testset):
  res=[]
  for item in testset:
    s=item[0]
    if s in vecs1.key_to_index and s in vecs2.key_to_index:
       sims = [1-vecs1.similarity(s, w) for w in item[1:] if w in vecs1.key_to_index and w in vecs2.key_to_index]
       res.extend(sims)
  return res

def cal_test_edit(vecs1, vecs2, testset):
  res=[]
  for item in testset:
    s=item[0]
    if s in vecs1.key_to_index and s in vecs2.key_to_index:
       edits = [edit_distance(s, w)/max(len(w), len(s)) for w in item[1:] if w in vecs1.key_to_index and w in vecs2.key_to_index]
       res.extend(edits)
  return res

def build_vocab(sentences, UNK='[UNK]', max_vocab=20000):

    print("building vocab...")
    step = 0
    wc = {}
    for sent in sentences:
        for word in sent:
            wc[word] = wc.get(word, 0) + 1               
    print("")
    idx2word = [UNK] + sorted(wc, key=wc.get, reverse=True)[:max_vocab]
    word2idx = {idx2word[idx]: idx for idx, _ in enumerate(idx2word)}
    vocab = set([word for word in word2idx])
    print("build done")

    print('number of words in vocab:',len(idx2word))
    assert len(idx2word) == len(word2idx.keys())

    return vocab, word2idx, idx2word

def build_char_vocab(idx2word, PAD='[PAD]'):
    print("\n\n Building character embedding model ....")
    #generate character vocab
    word2chars = {}
    char2id={PAD:0}
    for i in range(1,len(idx2word)):
        chars=[]
        word=idx2word[i]
        for char in list(word):
            if char in char2id:
                chars.append(char2id[char])
            else:
                char2id[char] = len(char2id)
                chars.append(char2id[char])
        word2chars[word] = torch.LongTensor(chars)
    return word2chars, char2id


#s_model w2v model for sentencces, ideally trained via gensim package
#c_model character-based w2v model for representing orthographic similarity
def create_test_set(sentences, s_model, c_model):
    #create FreqDist object
    unigram_dist = nltk.FreqDist(flatten(sentences))
    #first, convert the unigram_dist object to a pd.DataFrame object:
    unigrams_df = pd.DataFrame(pd.Series(unigram_dist))
    #rename the columns 
    unigrams_df.columns = ['counts']
    #Sort the values by count
    unigrams_df = unigrams_df.sort_values('counts', ascending=False)
    temp={w[0] for w in unigram_dist.most_common(5000)}
    temp=list(temp)[200:]
    random.shuffle(temp)
    test_words=temp[1:2000]
    test_pool=temp[2000:]

    test_set = []
    for word in test_words:
        if word == '[UNK]':
            continue
        _t=[word]
        pool=s_model.most_similar(word, topn=5)
        _t.extend([w[0] for w in pool])
        pool=c_model.most_similar(word, topn=5)
        _t.extend([w[0] for w in pool])
        _t.extend(random.sample(test_pool, 10))
        if len(_t) > 1:
            test_set.append(_t)

    return test_set



