import string

import nltk
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

sample = 'sample.txt'
file = open(sample)
sample = file.read()
file.close()

# Splitting by words
tokens = word_tokenize(sample)
# Converting to lower case
tokens = [w.lower() for w in tokens]
# Here we remove punctuation from each word
table = str.maketrans('', '', string.punctuation)
stripped = [w.translate(table) for w in tokens]
# Removing non alphabetic tokens
words = [word for word in stripped if word.isalpha()]
# Remove stop words here
stop_words = set(stopwords.words('english'))
words = [w for w in words if not w in stop_words]

words = ' '.join(words)

samplevec = [nltk.word_tokenize(words)]

print(samplevec[:5])

hyginus = Word2Vec(samplevec,
                   min_count=1,
                   sg=1,
                   vector_size=300,
                   epochs=30)
print("\n", hyginus.wv.most_similar('report', topn=10))
print(hyginus.wv['report'])

print(hyginus.wv.most_similar(positive=['forum'], topn=5))



print("\n", hyginus.wv.most_similar('report', topn=10))
print(hyginus.wv['report'])

print(hyginus.wv.most_similar(positive=['forum'], topn=5))
