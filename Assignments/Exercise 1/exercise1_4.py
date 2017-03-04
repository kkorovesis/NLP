######################################################### IMPORTS #########################################################
import nltk
import re
from nltk.tokenize import TweetTokenizer
from collections import Counter
import math

######################################################### FUNCTIONS #########################################################

def split_sequence(sequence, words):
    words = sequence.split()
    return words

def calculateUnigramProbLS(unigram, tokenized_corpus,  V):
    return ((tokenized_corpus.count(unigram) + 1)/(len(tokenized_corpus) + V))

def calculateBigramProbLS(bigram, final_corpus, final_corpus_bigrams, V):
    return ((final_corpus_bigrams.count(bigram) + 1)/(final_corpus.count(bigram[0]) + V))

def calculateTrigramsProbLS(trigram, final_corpus_trigrams, final_corpus_bigrams, V):
    first_uni = trigram[0]
    second_uni = trigram[1]
    bigram = (first_uni, second_uni)
    return ((final_corpus_trigrams.count(trigram) + 1)/ (final_corpus_bigrams.count(bigram) + V))

def estimateSingleWordProbabilityLS(sentence, unigrams, bigrams, bigrams_probs, unigrams_probs ):
    results = {}
    Probability = 1
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
    print (results)
    print (max(results, key=results.get))
    print (results.values())

#Markov Assumption
def estimateSentenceProbabilityLS(sentence, bigramed_sentence, unigrams, bigrams, bigrams_probs, unigrams_probs):
    Probability = 1
    first_bigram = bigramed_sentence[0]
    first_unigram = first_bigram[0]
    if first_unigram != "qwerty":
        i = 0;
        for unigram in unigrams:
            if unigram == first_unigram:
                Probability = Probability * unigrams_probs[i]
            i = i + 1
    for bigram in bigramed_sentence:
        if not re.search("qwerty", str(bigram)):
            i = 0
            for bb in bigrams:
                if bigram == bb:
                    Probability = Probability * bigrams_probs[i]
                i = i + 1
    print("P(", sentence, ") = ", Probability)

def estimateSentenceTrigramsProbabilityLS(sentence, trigramed_sentence, unigrams, bigrams, bigrams_probs, unigrams_probs, trigrams_probs):
    Probability = 1
    first_trigram = trigramed_sentence[0]
    first_unigram = first_trigram[0]
    first_bigram  = (first_trigram[0], first_trigram[1])
    if first_unigram != "qwerty":
        i = 0;
        for unigram in unigrams:
            if unigram == first_unigram:
                Probability = Probability * unigrams_probs[i]
            i = i + 1
    i = 0
    for bigram in bigrams:
        if not re.search("qwerty", str(bigram)):
            if bigram == first_bigram:
                Probability = Probability * bigrams_probs[i]
        i = i + 1
    for trigram in trigramed_sentence:
        if not re.search("qwerty", str(trigram)):
            i = 0
            for tr in trigrams:
                if tr == trigram:
                    Probability = Probability * trigrams_probs[i]
                i = i +1
    print("P(", sentence, ") = ", Probability)

######################################################### MAIN SCRIPT #########################################################


#Load Corpus and compute total bigrams
print ("Loading Corpus")
corpus = open('europarliamentENG.en', 'r').read()
tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
tokenized_corpus = tknzr.tokenize(corpus[0:100000])

#Replace words that appear less than 10 times in corpus
temp_counter = Counter(tokenized_corpus)
x = [word if temp_counter[word] >= 10 else 'qwerty' for word in tokenized_corpus]
x = ' '.join(x)

tokenized_corpus = tknzr.tokenize(x)
corpus_bigrams = list(nltk.ngrams(tokenized_corpus, 2))
print (corpus_bigrams)
print ("Corpus loaded with success!!!")

#Create Vocabulary
V = sorted(set(tokenized_corpus))
print ("Length of Vocabulary: ",(len(V)))



#Compute the probabilities for every Unigram!
print ("Training Bigram language Model...please wait...")
unigrams_probs = [0.0] * len(V)

i = 0
for unigram in V:
    if unigram != "qwerty":
        unigrams_probs[i] = calculateUnigramProbLS(unigram, tokenized_corpus, len(V) - 1)
    i = i + 1

i = 0;
for unigram in V:
    print ("P(",V[i],") = ", unigrams_probs[i])
    i = i + 1


#Compute the probabilities for every Bigram!
final_corpus_bigrams = list(nltk.ngrams(tokenized_corpus, 2))
bigrams = sorted(set(final_corpus_bigrams))
bigrams_probs = [0.0] * len(bigrams)


i = 0
for bigram in bigrams:
    if not re.search("qwerty", str(bigram)):
        bigrams_probs[i] = calculateBigramProbLS(bigram, tokenized_corpus, final_corpus_bigrams, len(V) - 1)
    i = i + 1

i = 0
for bigram in bigrams:
    print("P(", bigram, ") = ", bigrams_probs[i])
    i = i + 1


sentence = input("Please insert a sentence to test the Bigram Model: \n")
print ("Estimating Probability of given sentence and possible next words...")

word_sequence = []
seq1_words = split_sequence(sentence, word_sequence)
print (len(seq1_words))

if len(seq1_words) == 1:
    estimateSingleWordProbabilityLS(seq1_words, V, bigrams, bigrams_probs, unigrams_probs )
else:
    bigramed_sentence = list(nltk.ngrams(seq1_words, 2))
    estimateSentenceProbabilityLS(sentence, bigramed_sentence, V, bigrams, bigrams_probs, unigrams_probs)



#Compute Markov Assumption for the inserted sentence. Probabilities will be computed based on Laplace smoothing!
print ("Training Trigram language Model...please wait...")
final_corpus_trigrams = list(nltk.ngrams(tokenized_corpus, 3))
print (final_corpus_trigrams)
trigrams = sorted(set(final_corpus_trigrams))
trigrams_probs = [0.0] * len(trigrams)

count_querty_in_bigrams = 0
for bigram in bigrams:
    if re.search("querty", str(bigram)):
        count_querty_in_bigrams = count_querty_in_bigrams + 1

i = 0
for trigram in trigrams:
    if not re.search("qwerty", str(trigram)):
        trigrams_probs[i] = calculateTrigramsProbLS(trigram, final_corpus_trigrams, final_corpus_bigrams, len(bigrams) - count_querty_in_bigrams)
    i = i + 1

i = 0
for trigram in trigrams:
    print("P(", trigram, ") = ", trigrams_probs[i])
    i = i + 1

sentence = input("Please insert a sentence to test the Trigram Model <3: \n")
print ("Estimating Probability of given sentece and possible next words...")

word_sequence = []
seq1_words = split_sequence(sentence, word_sequence)

trigramed_sentence = list(nltk.ngrams(seq1_words, 3))
print (trigramed_sentence)
estimateSentenceTrigramsProbabilityLS(sentence, trigramed_sentence, V, bigrams, bigrams_probs, unigrams_probs, trigrams_probs)