import string

import matplotlib.pyplot
from sklearn.manifold import TSNE
import numpy as np
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
print(len(words))


samplevec = [nltk.word_tokenize(words)]

print(samplevec[:5])
hyginus = Word2Vec(samplevec,
                   window=15,
                   min_count=1,
                   workers=5,
                   negative=5,
                   sg=1,
                   vector_size=5,
                   epochs=5)
hyginus.save('hyginus')

re_hyginus = Word2Vec.load('hyginus')

print("Top 5 most similar words to forum: ", re_hyginus.wv.most_similar(positive='forum', topn=5))

print("The Similarity between average and highest is:", str(re_hyginus.wv.most_similar('average', 'highest')))


# Visualization


def reduce_dimensions(hyginus):
    num_dimensions = 2
    vectors = []
    labels = []
    for words in hyginus.wv.index_to_key:
        vectors.append(hyginus.wv[words])
        labels.append(words)

    # List to numpy
    vectors = np.asarray(vectors)
    # labels = np.asarray(labels)
    vectors = np.asarray(vectors)
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return x_vals, y_vals, labels


x_vals, y_vals, labels = reduce_dimensions(re_hyginus)


def plot_with_matplotlib(x_vals, y_vals, labels):
    import matplotlib.pyplot as plt
    import random

    random.seed(0)

    plt.figure(figsize=(12, 12))
    plt.scatter(x_vals, y_vals)

    # selected_indices = random.sample(indices, 25)
    selected_indices = []
    index = labels.index("highest")
    selected_indices.append(index)
    index = labels.index("fuel")
    selected_indices.append(index)
    index = labels.index("kia")
    selected_indices.append(index)
    index = labels.index("consumption")
    selected_indices.append(index)
    index = labels.index("cars")
    selected_indices.append(index)
    index = labels.index("average")
    selected_indices.append(index)
    index = labels.index("sold")
    selected_indices.append(index)
    index = labels.index("hyundai")
    selected_indices.append(index)

    for i in selected_indices:
        plt.annotate(labels[i], (x_vals[i], y_vals[i]))

        matplotlib.pyplot.savefig('SKIPGRAM.png')


plot_fn = plot_with_matplotlib

plot_fn(x_vals, y_vals, labels)
