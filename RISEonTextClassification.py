from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import matplotlib.pyplot as plt
import numpy as np

# Retrieve stopwords
stop_words = set(stopwords.words('english'))

# Define classification categories
categories = ['alt.atheism', 'comp.graphics', 'sci.med', 'sci.space', 'soc.religion.christian']

# Retrieve train set
twenty_train = fetch_20newsgroups(subset='train',
                                  remove=('headers', 'footers', 'quotes'),
                                  categories=categories,
                                  shuffle=True,
                                  random_state=42)

print("The following newsgroups will be tested: {}".format(twenty_train.target_names))

# Stochastic Gradient Descent Classifier
model = Pipeline([('vect', CountVectorizer()),
                  ('tfidf', TfidfTransformer()),
                  ('clf', SGDClassifier(loss='modified_huber',
                                        penalty='l2',
                                        alpha=1e-3,
                                        max_iter=50,
                                        tol=0,
                                        random_state=42))])

# model = Pipeline([('vect', CountVectorizer()),
#                   ('tfidf', TfidfTransformer()),
#                   ('clf', MultinomialNB())])

# Training the Classifier
model.fit(twenty_train.data, twenty_train.target)


# Retrieve test set
twenty_test = fetch_20newsgroups(subset='test',
                                 remove=('headers', 'footers', 'quotes'),
                                 categories=categories,
                                 shuffle=True,
                                 random_state=42)


# Remove all the stopwords (as defined in 'stop_words') from 'text'.
def remove_stopwords(text):
    word_tokens = word_tokenize(text)
    filtered_words = [w for w in word_tokens if w not in stop_words]
    return filtered_words


# Mask all tokens equal to 'mask_word' in 'text'.
def mask(text, mask_word):
    return text.replace(mask_word, ' ')


# Predicting the test set
# for nb_test_example in range(len(twenty_test.data)):
for nb_test_example in range(25):

    # Prediction for original text
    original_input = twenty_test.data[nb_test_example]
    predicted = model.predict_proba([original_input])
    initial_certainty = predicted[0][twenty_test.target[nb_test_example]]

    # Dictionary holding the certainty gain for every word
    words_certainty = {}

    # Predictions for masked texts
    for word in remove_stopwords(original_input):
        modified_input = mask(twenty_test.data[nb_test_example], word)
        predicted = model.predict_proba([modified_input])

        certainty = predicted[0][twenty_test.target[nb_test_example]]

        words_certainty[word] = (initial_certainty - certainty)

    # Sort the words by their importance
    words_sorted_by_certainty = sorted(words_certainty.items(), key=lambda kv: kv[1], reverse=True)

    # Top n words to show
    n = 10
    n = min(n, len(words_sorted_by_certainty))

    # Plot
    plt.close()
    plt.title("The most relevant words to predict file "
              + twenty_test.filenames[nb_test_example].rsplit('\\', 1)[1]
              + " as " + categories[twenty_test.target[nb_test_example]]
              + " (its correct category) with probability "
              + str(round(initial_certainty, 2)) + " are:", wrap=True)
    x = np.arange(n)
    plt.bar(x, height=[a for (_, a) in words_sorted_by_certainty][:n])
    plt.xticks(x, [a for (a, _) in words_sorted_by_certainty][:n])
    plt.xlabel("word")
    plt.ylabel("certainty gain")
    plt.show()
