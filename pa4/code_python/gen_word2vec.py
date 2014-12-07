#use word2vec to generate word vector for NLP pa4
from gensim.models import Word2Vec
import word2vec as w2vwrapper
import numpy as Math

model = Word2Vec.load_word2vec_format('/Users/enhao/Documents/Courses/Fall_2014/CS224N_NLP/pa4/vectors.bin', binary=True)  # C binary format

X = Math.loadtxt("/Users/enhao/Documents/Courses/Fall_2014/CS224N_NLP/pa4/wordvector_after.txt");

f_vocab = open('/Users/enhao/Documents/Courses/Fall_2014/CS224N_NLP/pa4/vocab.txt', 'r')
f_vector = open('/Users/enhao/Documents/Courses/Fall_2014/CS224N_NLP/pa4/vector_word2vec.txt', 'w')
for line in f_vocab:
    tmp_vocab=line
    if (tmp_vocab[-1:]=='\n'):
        tmp_vocab=tmp_vocab[:-2]
    try:
        tmp_feature=model[tmp_vocab]
    except KeyError, e:
        tmp_feature=model['nan']
    for v in tmp_feature:
        f_vector.write('{0} '.format(v))
    f_vector.write('\n')
f_vocab.close()
f_vector.close()
