from nltk import ngrams
import re
import string

def ngrams_estimation(text, ngram_range=(2, 2)):
    """
    Estimate the n-grams of the text
    :param      text: string/list, required
                    -The given text
    :param      ngram_range: tuple, optional
                    -Range of grams
    :return:    list
                    -The list contains the total number of available ngrams
    """
    list_of_ngrams = []
    if not isinstance(text, list):
        text = [text]
    for i in range(ngram_range[0], ngram_range[1] + 1):
        for sentence in text:
            n_grams = [' '.join(ngram) for ngram in ngrams(sentence.split(), i)]
            list_of_ngrams.extend(n_grams)
    return list_of_ngrams

def remove_punc(text):
    return re.sub(r'[{}]'.format('\\'.join(string.punctuation)), ' ', text)

