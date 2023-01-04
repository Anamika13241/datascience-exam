import nltk
nltk.download()
from nltk.util import ngrams
a="sussi is a  good girl"
NGrams=ngrams(sequence=nltk.word_tokenize(a),n=3)
for i in NGrams:
    print(i)
