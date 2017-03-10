######################################################### IMPORTS #########################################################
import nltk, re, codecs, tools, string
from math import log
from nltk.tokenize import TweetTokenizer
from collections import Counter

######################################################### FUNCTIONS #########################################################

def remove_punc(text):
    return re.sub(r'[{}]'.format('\\'.join(string.punctuation)), ' ', text)

def split_sequence(sequence, words):
    words = sequence.split()
    return words

def calculateUnigramProbLS(unigram, tokenized_corpus,  V):
    return (tokenized_corpus.count(unigram) + 1)/(len(tokenized_corpus) + V)

def calculateBigramProbLS(bigram, final_corpus, final_corpus_bigrams, V):
    return (final_corpus_bigrams.count(bigram) + 1)/(final_corpus.count(bigram[0]) + V)

# def calculateUnigramLogProbLS(unigram, tokenized_corpus,  V):
#     return log((tokenized_corpus.count(unigram) + 1) / (len(tokenized_corpus) + V))
#
# def calculateBigramLogProbLS(bigram, final_corpus, final_corpus_bigrams, V):
#     return log((final_corpus_bigrams.count(bigram) + 1) / (final_corpus.count(bigram[0]) + V))

def estimateNextWordProbability(sentence, unigrams, bigrams, bigrams_probs, unigrams_probs ):
    results = {}
    Probability = 1
    if len(sentence) == 1:
        the_unigram = sentence[0]
        if the_unigram != "qwerty":
            i = 0;
            for unigram in unigrams:
                if unigram == the_unigram:
                    Probability = Probability * unigrams_probs[i]
                i = i + 1
        i = 0
        for bigram in bigrams:
            if not re.search("qwerty", str(bigram)):
                if bigram[0] == the_unigram:
                    # Probability = Probability_unigram * bigrams_probs[i]
                    print("P(", bigram, ") = ", Probability * bigrams_probs[i])
                    results[bigram[1]] = Probability * bigrams_probs[i]
            i = i + 1
        if len(results) > 0:
            print("Most possible next word: ", max(results, key=results.get))
            print("Possible next words and their probabilities to appear: ", results)
        else:
            print("Could not predict the next word for your given sentece...Does your word exists?")

    elif len(sentence) >= 2:
        last_bigram = bigramed_sentence[len(bigramed_sentence) - 1]
        print(last_bigram)
        last_unigram = last_bigram[1]
        i=0
        for bigram in bigrams:
            if not re.search("qwerty", str(bigram)):
                if bigram[0] == last_unigram:
                    results[bigram[1]] = Probability * bigrams_probs[i]
            i = i + 1
        if len(results) > 0:
            print("Most possible next word: ", max(results, key=results.get))
            print("Possible next words and their probabilities to appear: ", results)
        else:
            print("Could not predict the next word for your given sentece...Does your word exists?")

# Markov Assumption
def estimateSentenceProbabilityLS(sentence, bigramed_sentence, unigrams, bigrams, bigrams_probs, unigrams_probs):
    Probability = 1
    first_bigram = bigramed_sentence[0]  ##Need to make it work for full sentences and not only for one bigram##
    first_unigram = first_bigram[0]
    if first_unigram != "qwerty":
        i = 0;
        for unigram in unigrams:
            if unigram == first_unigram:
                Probability = Probability * unigrams_probs[i]
            i = + 1
    for bigram in bigramed_sentence:
        if not re.search("qwerty", str(bigram)):
            i = 0
            for bb in bigrams:
                if bigram == bb:
                    Probability = Probability * bigrams_probs[i]
                i=+ 1
    print("P(", sentence, ") = ", Probability)

# Cross-entropy Calculator
def estimateLanguageCrossEntropy(bigrams, bigrams_probs):
    Slog = 0
    i = 0
    for bigram in bigrams:
        if not re.search("qwerty", str(bigram)):
            Slog += log(bigrams_probs[i], 2)
        i=+1
    Slog = -(1.0 / len(final_corpus_bigrams)) * Slog

    return Slog


######################################################### MAIN SCRIPT ################################################

#Load Corpus and compute total bigrams
print ("Loading Corpus")
corpus = codecs.open(r'C:\Corpus\europarl-v7.fr-en.en', 'r', encoding='utf-8', errors='ignore').read()
print ("Corpus loaded with success!!! Length: ", len(corpus))

#Initialize tokenization method
tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
print ("Processing corpus... please wait!")
corpus = corpus.lower()
corpus = remove_punc(corpus)
tokenized_corpus = tknzr.tokenize(corpus[0:100000])

#Replace words that appear less than 10 times in corpus
temp_counter = Counter(tokenized_corpus)
x = [word if temp_counter[word] >= 10 else 'qwerty' for word in tokenized_corpus]
x = ' '.join(x)

tokenized_corpus = tknzr.tokenize(x)
#Create Vocabulary
V = sorted(set(tokenized_corpus))

print ("Training Bigram language Model... please wait...")

#Compute the probabilities for every Unigram!
unigrams_probs = [0.0] * len(V)

i = 0
for unigram in V:
    if unigram != "qwerty":
        unigrams_probs[i] = calculateUnigramProbLS(unigram, tokenized_corpus, len(V) - 1)
    i = i + 1

f = open(r'output_files\UnigramsProbabilities.txt', 'w')
i = 0;
for unigram in V:
   # print ("P(",V[i],") = ", unigrams_probs[i])
    f.write("P(" + (V[i]) + ") = " + str(unigrams_probs[i]) + "\n")
    i = i + 1
f.close()

#Compute the probabilities for every Bigram!
final_corpus_bigrams = list(nltk.ngrams(tokenized_corpus, 2))
bigrams = sorted(set(final_corpus_bigrams))
bigrams_probs = [0.0] * len(bigrams)

i = 0
for bigram in bigrams:
    if not re.search("qwerty", str(bigram)):
        bigrams_probs[i] = calculateBigramProbLS(bigram, tokenized_corpus, final_corpus_bigrams, len(V) - 1)
    i = i + 1

f = open(r'output_files\BigramsProbabilities.txt', 'w')
i = 0
for bigram in bigrams:
    #print("P(", bigram, ") = ", bigrams_probs[i])
    f.write("P(" + str(bigram) + ") = " + str(bigrams_probs[i]) + "\n")
    i = i + 1

# ####Logarithic Probabilities#####
#
# #Compute the log probabilities for every Unigram!
# unigrams_log_probs = [0.0] * len(V)
#
# i = 0
# for unigram in V:
#     if unigram != "qwerty":
#         unigrams_log_probs[i] = calculateUnigramLogProbLS(unigram, tokenized_corpus, len(V) - 1)
#     i = i + 1
#
# f = open(r'output_files\UnigramsLogProbabilities.txt', 'w')
# i = 0;
# for unigram in V:
#    # print ("P(",V[i],") = ", unigrams_probs[i])
#     f.write("P(" + (V[i]) + ") = " + str(unigrams_log_probs[i]) + "\n")
#     i = i + 1
# f.close()
#
# #Compute the log probabilities for every Bigram!
# final_corpus_bigrams = list(nltk.ngrams(tokenized_corpus, 2))
# bigrams = sorted(set(final_corpus_bigrams))
# bigrams_log_probs = [0.0] * len(bigrams)
#
# i = 0
# for bigram in bigrams:
#     if not re.search("qwerty", str(bigram)):
#         bigrams_log_probs[i] = calculateBigramLogProbLS(bigram, tokenized_corpus, final_corpus_bigrams, len(V) - 1)
#     i = i + 1
#
# f = open(r'output_files\BigramsLogProbabilities.txt', 'w')
# i = 0
# for bigram in bigrams:
#     #print("P(", bigram, ") = ", bigrams_probs[i])
#     f.write("P(" + str(bigram) + ") = " + str(bigrams_log_probs[i]) + "\n")
#     i = i + 1

# Languge Model Test
input_sentence = input("Please insert a sentence to test the Bigram Model: \n")
print ("Processing Input Sentece... please wait!")

# Initialize tokenization method
input_sentence = input_sentence.lower()
input_sentence = remove_punc(input_sentence)
tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
tokenized_input_sentence = tknzr.tokenize(input_sentence)
input_sentence_bigrams = list(nltk.ngrams(tokenized_input_sentence, 2))
# print (input_sentence_bigrams)

print ("Testing Model...")
print(input_sentence_bigrams)
estimateSentenceProbabilityLS(input_sentence, input_sentence_bigrams, V, bigrams, bigrams_probs, unigrams_probs)

# Estimating next word
sentence = input("Please insert a sentence to test the Bigram Model: \n")
sentence = sentence.lower()
print ("Estimating Probability of given sentence and possible next words...")

word_sequence = []
seq1_words = split_sequence(sentence, word_sequence)

bigramed_sentence = []

if len(seq1_words) == 1:
    estimateNextWordProbability(seq1_words, V, bigrams, bigrams_probs, unigrams_probs )
elif len(seq1_words) >= 2:
    bigramed_sentence = list(nltk.ngrams(seq1_words, 2))
    estimateSentenceProbabilityLS(bigramed_sentence, V, bigrams, bigrams_probs, unigrams_probs)
    estimateNextWordProbability(seq1_words, V, bigrams, bigrams_probs, unigrams_probs)

#Compute Cross-Entropy
print("Language Model cross entropy for bigrams: " + str(estimateLanguageCrossEntropy(bigrams, bigrams_probs)))

