########################################################################## IMPORTS ##########################################################################

import nltk, re, string, math, codecs
from nltk.tokenize import TweetTokenizer
from collections import Counter

########################################################################## FUNCTIONS ##########################################################################

def remove_punc(text):
    print("Removing punctuations from the corpus...")
    return re.sub(r'[{}]'.format('\\'.join(string.punctuation)), ' ', text)

def split_sequence(sequence, words):
    words = sequence.split()
    return words

def calculateUnigramProbLS(unigram, tokenized_corpus, V):
    return (math.log((tokenized_corpus.count(unigram) + 1)/(len(tokenized_corpus) + V), 10))

def calculateBigramProbLS(bigram, final_corpus, final_corpus_bigrams, V):
    return (math.log((final_corpus_bigrams.count(bigram) + 1)/(final_corpus.count(bigram[0]) + V), 10))

def calculateTrigramProbLS(trigram, final_corpus_trigrams, final_corpus_bigrams, Voc):
    return (math.log((final_corpus_trigrams.count(trigram) + 1)/(final_corpus_bigrams.count((trigram[0], trigram[1])) + Voc), 10))

def countRareUnigrams(unigrams):
    count = 0
    for unigram in unigrams:
        if unigram == "qwerty":
            count =  count + 1
    return count

def countRareBigrams(bigrams):
    count = 0
    for bigram in bigrams:
        if re.search("qwerty", str(bigram)):
            count = count + 1
    return count

def countRareTrigrams(trigrams):
    count = 0
    for trigram in trigrams:
        if re.search("qwerty", str(trigram)):
            count = count + 1
    return count

def estimateNextWordProbability_BiLM(sentence, unigrams, bigrams, bigrams_probs, unigrams_probs):
    found = 0
    results = {}
    Probability = 0
    if len(sentence) == 1:
        the_unigram = sentence[0]
        if the_unigram != 'qwerty':
            i = 0
            for unigram in unigrams:
                if unigram == the_unigram:
                    Probability = Probability + unigrams_probs[i]
                    found = 1
                i = i + 1
        if found == 1:
            i = 0
            for bigram in bigrams:
                if not re.search("qwerty", str(bigram)):
                    if bigram[0] == the_unigram:
                      #  print("P(", bigram , ") = ", Probability * bigrams_probs[i])
                        results[bigram[i]] = Probability + bigrams_probs[i]
                i = i + 1
            if len(results) > 0:
                print("Most possible next word: ", max(results, key=results.get))
                print("Possible next words and their probabilities to appear: ", results)
                print(" ")
            else:
                print("Could not predict the next word for your given sentence... :( \n")
        else:
            print("Could not predict the next word for your given sentence... :( \n")
    elif len(sentence) >= 2:
        last_bigram = bigramed_sentence[len(bigramed_sentence) - 1]
        last_unigram = last_bigram[1]
        if last_unigram != 'qwerty':
            i = 0
            for unigram in unigrams:
                if unigram == last_unigram:
                    Probability = Probability + unigrams_probs[i]
                    found = 1
                i = i + 1
        if found == 1:
            i = 0
            for bigram in bigrams:
                if not re.search("qwerty", str(bigram)):
                    if bigram[0] == last_unigram:
                      #  print("P(", bigram, ") = ", Probability * bigrams_probs[i])
                        results[bigrams[i]] = Probability + bigrams_probs[i]
                i = i + 1
            if len(results) > 0:
                print("Most possible next word: ", max(results, key=results.get))
                print("Possible next words and their probabilities to appear: ", results)
                print(" ")
            else:
                print("Could not predict the next word for your given sentence... :( \n")
        else:
            print("Could not predict the next word for your given sentence... :( \n")

def estimateSentenceProbabilityLS_BiLM(sentence, bigramed_sentence, unigrams, bigrams, bigrams_probs, unigrams_probs, tokenized_corpus):
    Probability = 0
    first_bigram = bigramed_sentence[0]
    first_unigram = first_bigram[0]
    if first_unigram != "qwerty":
        unigram_found = 0
        i = 0
        for unigram in unigrams:
            if unigram == first_unigram:
                Probability = Probability + unigrams_probs[i]
                #print("First unigram: ",Probability)
                unigram_found = 1
            i = i + 1
    if unigram_found == 0:
        #print(unigram_found)
        prob = calculateUnigramProbLS(first_unigram, tokenized_corpus, len(unigrams) - 1)
        Probability = Probability + prob
    for bigram in bigramed_sentence:
        bigram_found = 0
        if not re.search("qwerty", str(bigram)):
            i = 0
            for bb in bigrams:
                if bigram == bb:
                    Probability = Probability + bigrams_probs[i]
                    #print(bb,"     ",Probability)
                    bigram_found = 1
                    #print(bigram_found)
                i = i + 1
        if bigram_found == 0:
            #print(bigram_found)
            Probability = Probability + calculateBigramProbLS(bigram, tokenized_corpus, final_corpus_bigrams, len(unigrams) - 1)
            #print("not_found ", calculateBigramProbLS(bigram, tokenized_corpus, final_corpus_bigrams, len(V) - 1))
    #print("Probability  ", Probability)
    print ("P(", sentence, ") = ", Probability)

def estimateSentenceProbabilityLS_TriLM(sentence, trigramed_sentence, unigrams ,unigrams_probs, bigrams, bigrams_probs, trigrams, trigrams_probs, tokenized_corpus):
    first_trigram = trigramed_sentence[0]
    first_bigram  = (first_trigram[0], first_trigram[1])
    first_unigram = first_trigram[0]
    Probability = 0
    if first_unigram != "qwerty":
        unigram_found = 0
        i = 0
        for unigram in unigrams:
            if unigram == first_unigram:
                Probability = Probability + unigrams_probs[i]
                #print("First unigram: ",Probability)
                unigram_found = 1
            i = i + 1
    if unigram_found == 0:
        #print(unigram_found)
        prob = calculateUnigramProbLS(first_unigram, tokenized_corpus, len(unigrams) - 1)
        Probability = Probability + prob
    i = 0
    bigram_found = 0
    for bigram in bigrams:
        if bigram == first_bigram:
            Probability = Probability + bigrams_probs[i]
            bigram_found = 1
        i = i + 1
    if bigram_found == 0:
        Probability = Probability + calculateBigramProbLS(bigram, tokenized_corpus, final_corpus_bigrams, len(unigrams) - 1)
    for trigram in trigramed_sentence:
        trigram_found = 0
        if not re.search("qwerty", str(trigram)):
            i = 0
            for tr in trigrams:
                if trigram == tr:
                    Probability = Probability + trigrams_probs[i]
                    #print(bb,"     ",Probability)
                    trigram_found = 1
                    #print(bigram_found)
                i = i + 1
        if trigram_found == 0:
            #print(bigram_found)
            Probability = Probability + calculateTrigramProbLS(trigram, tokenized_corpus, final_corpus_bigrams, len(bigrams) - countRareBigrams(bigrams))
            #print("not_found ", calculateBigramProbLS(bigram, tokenized_corpus, final_corpus_bigrams, len(V) - 1))
    print("P(", sentence, ") = ", Probability)

def estimateNextWordProbability_TriLM(trigramed_sentence, unigrams, unigrams_probs, trigrams, trigrams_probs):
    results = {}
    last_trigram = trigramed_sentence[len(trigramed_sentence) - 1]
    last_unigram = last_trigram[2]
    Probability = 0
    if last_unigram != 'qwerty':
        i = 0
        for unigram in unigrams:
            if unigram == last_unigram:
                Probability = Probability + unigrams_probs[i]
                found = 1
            i = i + 1
    if found == 1:
        i = 0
        for trigram in trigrams:
            if not re.search("qwerty", str(trigram)):
                if trigram[0] == last_unigram:
                    results[trigrams[i]] = Probability + trigrams_probs[i]
            i = i + 1
        if len(results) > 0:
            print("Most possible next word: ", max(results, key=results.get))
            print("Possible next words and their probabilities to appear: ", results)
            print(" ")
        else:
            print("Could not predict the next word for your given sentence... :( \n")
    else:
        print("Could not predict the next word for your given sentence... :( \n")

def estimateLanguageCrossEntropy_BiLM(test_corpus, tokenized_corpus, final_corpus_bigrams, unigrams, bigrams, bigrams_probs):
    tkr = TweetTokenizer(strip_handles=True, reduce_len=True)
    test_corpus = test_corpus.lower()
    tokenized_test_corpus = tkr.tokenize(test_corpus[300000:350000])
    test_bigrams = list(nltk.ngrams(tokenized_test_corpus, 2))
    Slog = 0
    for test_bigram in test_bigrams:
        i = 0
        test_bigram_found = 0
        for train_bigram in bigrams:
            if (test_bigram == train_bigram):
                Slog = Slog + bigrams_probs[i]
                test_bigram_found = 1
            i = i + 1
        if test_bigram_found == 0:
            Slog = Slog + calculateBigramProbLS(test_bigram, tokenized_corpus, final_corpus_bigrams, len(unigrams) - 1)
    print ("Cross-Entropy: ", -(Slog/len(tokenized_test_corpus)))
    calculatePerplexity(Slog, len(tokenized_test_corpus))

def estimateLanguageCrossEntropy_TriLM(test_corpus, tokenized_corpus, final_corpus_bigrams, bigrams, trigrams, trigrams_probs):
    tkr = TweetTokenizer(strip_handles=True, reduce_len=True)
    test_corpus = test_corpus.lower()
    tokenized_test_corpus = tkr.tokenize(test_corpus[300000:350000])
    test_trigrams = list(nltk.ngrams(tokenized_test_corpus, 3))
    Slog = 0
    for test_trigram in test_trigrams:
        i = 0
        test_trigram_found = 0
        for train_trigram in trigrams:
            if (test_trigram == train_trigram):
                Slog = Slog + trigrams_probs[i]
                test_trigram_found = 1
            i = i + 1
        if test_trigram_found == 0:
            Slog = Slog + calculateTrigramProbLS(test_trigram, tokenized_corpus, final_corpus_bigrams, len(bigrams) - countRareBigrams(bigrams))
    print ("Cross-Entropy: ", -(Slog/len(tokenized_test_corpus)))
    calculatePerplexity(Slog, len(tokenized_test_corpus))

def calculatePerplexity(Slog, N):
    # print ("Perplexity: ", pow(1/Slog, 1/float(N)))
    print ("Perplexity: ", ((1/Slog)** (1/float(N))))

########################################################################## MAIN SCRIPT ##########################################################################

#Load corpus
corpus_name = input("Enter Corpus for Training: \n")
print("Loading Corpus ", corpus_name)
# corpus = open(corpus_name, 'r').read()
corpus = codecs.open(corpus_name, 'r', encoding='utf-8', errors='ignore').read()
print("Corpus loaded with success!!! Length ", len(corpus))
test_corpus_name = input("Enter Corpus for Testing: \n")
print("Loading test Corpus ", test_corpus_name)
test_corpus = codecs.open(corpus_name, 'r', encoding='utf-8', errors='ignore').read()
# test_corpus = open(test_corpus_name, 'r').read()
print("Corpus for testing loaded with success!!! Length ", len(test_corpus))

#Initialize tokenization method and process Corpus
tknzr = TweetTokenizer(strip_handles= True, reduce_len= True)
print("Processing Corpus for Training...")
corpus = corpus.lower()
tokenized_corpus = tknzr.tokenize(corpus[0:100000])

#Replace rare words
counter = Counter(tokenized_corpus)
x = [word if counter[word] >= 10 else 'qwerty' for word in tokenized_corpus]
x = ' '.join(x)

#Training LMs
print("Training the Language Models. Please be patient, it may take a few minutes (depending on Corpus size)")
tokenized_corpus = tknzr.tokenize(x)
corpus_bigrams = list(nltk.ngrams(tokenized_corpus, 2))
V = sorted(set(tokenized_corpus))

unigram_probs = [0.0] * len(V)

i = 0
for unigram in V:
    if unigram != 'qwerty':
        unigram_probs[i] = calculateUnigramProbLS(unigram, tokenized_corpus, len(V) - 1)

    i = i + 1

f = open('UnigramsProbabilities.txt', 'w')
i = 0
for unigram in V:
    f.write("P(" + (V[i]) + ") = " + str(unigram_probs[i]) + "\n")
    i = i + 1
f.close()

final_corpus_bigrams = list(nltk.ngrams(tokenized_corpus, 2))
bigrams = sorted(set(final_corpus_bigrams))
bigrams_probs = [0.0] * len(bigrams)

i = 0
for bigram in bigrams:
    if not re.search("qwerty" , str(bigram)):
        bigrams_probs[i] = calculateBigramProbLS(bigram, tokenized_corpus, final_corpus_bigrams, len(V) - 1)
    i = i + 1


f = open('BigramsProbabilities.txt', 'w')
i = 0
for bigram in bigrams:
    f.write("P(" + str(bigram) + ") = " + str(bigrams_probs[i]) + "\n")
    i = i + 1
f.close()

final_corpus_trigrams = list(nltk.ngrams(tokenized_corpus, 3))
trigrams = sorted(set(final_corpus_trigrams))
trigrams_probs = [0.0] * len(trigrams)

i = 0
for trigram in trigrams:
    if not re.search("qwerty", str(trigram)):
        trigrams_probs[i] = calculateTrigramProbLS(trigram, final_corpus_trigrams, final_corpus_bigrams, len(bigrams) - countRareBigrams(bigrams))
    i = i + 1
f = open('TrigramsProbabilities.txt', 'w')
i = 0
for trigram in trigrams:
    f.write("P(" + str(trigram) + ") = " + str(trigrams_probs[i]) + "\n")
    i = i + 1

sentence = input("Please insert a sentence to test the LMs: \n")
sentence = sentence.lower()
print("Estimating Probability of given sentence and possible next word \n")

word_sequence = []
seq1_words = split_sequence(sentence, word_sequence)

print("")
print("######################################################## Bigram Language Model ########################################################")
bigramed_sentence = []
if (len(seq1_words) == 1):
    estimateNextWordProbability_BiLM(seq1_words, V, bigrams, bigrams_probs, unigram_probs)
elif (len(seq1_words) >= 2):
    bigramed_sentence = list(nltk.ngrams(seq1_words, 2))
    estimateSentenceProbabilityLS_BiLM(sentence, bigramed_sentence, V, bigrams, bigrams_probs, unigram_probs, tokenized_corpus)
    estimateNextWordProbability_BiLM(seq1_words, V, bigrams, bigrams_probs, unigram_probs)

estimateLanguageCrossEntropy_BiLM(test_corpus, tokenized_corpus, final_corpus_bigrams, V, bigrams, bigrams_probs)

print("")
print("######################################################## Trigram Language Model ########################################################")
trigramed_sentence = []
trigramed_sentence = list(nltk.ngrams(seq1_words, 3))
estimateSentenceProbabilityLS_TriLM(sentence, trigramed_sentence, V, unigram_probs, bigrams, bigrams_probs, trigrams, trigrams_probs, tokenized_corpus)
estimateNextWordProbability_TriLM(trigramed_sentence, V, unigram_probs, trigrams, trigrams_probs)

estimateLanguageCrossEntropy_TriLM(test_corpus, tokenized_corpus, final_corpus_bigrams, bigrams, trigrams, trigrams_probs)