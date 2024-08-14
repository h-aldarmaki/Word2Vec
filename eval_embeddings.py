from gensim.models import KeyedVectors
from gensim import models
import utils
from nltk.tokenize import word_tokenize
from scipy import stats

filename ='librispeech_100.txt'
source_vecs='out/sgns_C_large/words.vec'

#process dataset
file = open(filename, 'rt')
text = file.readlines()
file.close()

n_s=0
n_w=0
word2idx = {}
idx2word = {}
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
            if word not in word2idx:
              word2idx[word] = word_idx
              idx2word[word_idx]=word
              word_idx+=1
  s_sentences.append(new_sen)
  n_s+=1

print("Number of sentences:", n_s)
print("Number of words:", n_w)
print("Number of words in vocab:", word_idx)


#load c model
c_model = models.KeyedVectors.load_word2vec_format('char_embeddings.vec')

#train gensim target model
print("training target model ... ")
t_model=models.Word2Vec(s_sentences, epochs=100)

#load source model
s_model = models.KeyedVectors.load_word2vec_format(source_vecs)

#create test set
print("creating test set ... ")
test_set=utils.create_test_set(s_sentences, t_model.wv, c_model)

#eval source embeddings
print("calculating similarities ... ")
X=utils.calc_test_sim(s_model, t_model.wv, test_set)
Y=utils.calc_test_sim(t_model.wv, s_model,  test_set)
Z=utils.cal_test_edit(t_model.wv, s_model, test_set)
correlation, p_value = stats.pearsonr(X, Y)
print("Pearson correlation w. target model sim sim:", correlation)
correlation, p_value = stats.pearsonr(X, Z)
print("Pearson correlation w. edit distance:", correlation)

print ("\nExamples of nearest neighbors:")
for word in ['monday', 'sun', 'green', 'walk', 'happy', 'think', 'quickly', 'walking', 'one']:
  print(word, s_model.most_similar(word, topn=5))