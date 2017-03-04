######################################################### IMPORTS #########################################################
import nltk
import re

######################################################### FUNCTIONS #########################################################

def split_sequence(sequence, words):
    words = sequence.split()
    return words
    #for word in words:
    #    print(word)

def calculateUnigramProbLS(unigram, final_corpus,  V):
    return (final_corpus.count(unigram) + 1)/(len(final_corpus) + len(V))

def calculateBigramProbLS(bigram, final_corpus, final_corpus_bigrams, V):
    return (final_corpus_bigrams.count(bigram) + 1)/(final_corpus.count(bigram[0]) + len(V))

def estimateSentenceProbabilityLS(sentence, bigramed_sentence, unigrams, bigrams, bigrams_probs, unigrams_probs):
    Probability = 1
    first_bigram = bigramed_sentence[0]
    first_unigram = first_bigram[0]
    i = 0;
    for unigram in unigrams:
        if unigram == first_unigram:
            #print (unigram , " ", Probability)
            Probability = Probability * unigrams_probs[i]
        i = i + 1
    for bigram in bigramed_sentence:
        i = 0
        for bb in bigrams:
            if bigram == bb:
                #print(bigram, " ", bigrams_probs[i])
                Probability = Probability * bigrams_probs[i]
            i = i + 1
    print("P(", sentence, ") = ", Probability)


######################################################### MAIN SCRIPT #########################################################

seq1_words = []
seq2_words = []
seq3_words = []
seq4_words = []
seq5_words = []

word_sequence1 = "He please god football"
word_sequence2 = "He plays god football"
word_sequence3 = "He plays good football"
word_sequence4 = "He players good football"
word_sequence5 = "He pleases god ball"

seq1_words = split_sequence(word_sequence1, seq1_words)
seq2_words = split_sequence(word_sequence2, seq2_words)
seq3_words = split_sequence(word_sequence3, seq3_words)
seq4_words = split_sequence(word_sequence4, seq4_words)
seq5_words = split_sequence(word_sequence5, seq5_words)

print ("Loading Corpus (:Euro Parliament in English)")

#Starting Corpus (Text we got from the Web)
corpus = open('europarliamentENG.en', 'r').read()
#Clean the starting Corpus from various characters
cleaned_corpus = re.sub('\W+', ' ' , corpus)

#Rerieve tokens as the final corpus
final_corpus = cleaned_corpus.split(" ")

print ("Corpus loaded with success!!!")
print ("Length of starting Corpus (text):  ", (len(corpus)))
print ("Length of cleaned Corpus: (text)", (len(cleaned_corpus)))
print ("Length of tokenized Corpus: (tokens:with duplicates)", (len(final_corpus)))


#Compute Vocabulary and its size
#Get the final Vocabulary by deleting duplicated from final corpus
#V is the number of unique (n-1)grams you have in the corpus (In case of Laplace smoothing)
V = sorted(set(final_corpus))
print ("Length of Vocabulary: ",(len(V)))

#Calculate lexical richness of the text
print ("Lexical richness of Corpus: ", len(V) / len(final_corpus))


#Estimate the probabilities of all the word bigrams of the five candidate word sequences above
seq1_bigrams = list(nltk.bigrams(seq1_words))
seq2_bigrams = list(nltk.bigrams(seq2_words))
seq3_bigrams = list(nltk.bigrams(seq3_words))
seq4_bigrams = list(nltk.bigrams(seq4_words))
seq5_bigrams = list(nltk.bigrams(seq5_words))

#First compute the probability for single terms
temp_seq1 = word_sequence1 + " " +word_sequence2 + " " + word_sequence3 + " " + word_sequence4 + " " +  word_sequence5
temp_seq2 = temp_seq1.split(" ")
unigrams  = sorted(set(temp_seq2))

temp_count = 0;
unigrams_probs = [0.0] * len(unigrams)

print (" ")
print ("Computing probabilities of the unigrams...")
i = 0
for unigram in unigrams:
    unigrams_probs[i] = calculateUnigramProbLS(unigram, final_corpus, V)
    i = i + 1

i = 0;
for unigram in unigrams:
    print ("P(",unigrams[i],") = ", unigrams_probs[i])
    i = i + 1




print (" ")
print (" ")



bigrams = seq1_bigrams + seq2_bigrams + seq3_bigrams + seq4_bigrams + seq5_bigrams
bigrams = sorted(set(bigrams))

bigrams_probs = [0.0] * len(bigrams)


#compute all possible bigrams of the final corpus
print ("Calculating all possible bigrams inside our corpus...Please wait...")
final_corpus_bigrams = list(nltk.bigrams(final_corpus))

# Compute the probabilities using Laplace smoothing
print ("\nComputing probabilities of the bigrams...")

i = 0
for bigram in bigrams:
    bigrams_probs[i] = calculateBigramProbLS(bigram, final_corpus, final_corpus_bigrams, V)
    i = i + 1

i = 0
for bigram in bigrams:
    print("P(", bigram, ") = ", bigrams_probs[i])
    i = i + 1


print(" ")
print("Estimating the probability for each one of the given sentences, based on the above calculations...")

estimateSentenceProbabilityLS(word_sequence1, seq1_bigrams, unigrams, bigrams, bigrams_probs, unigrams_probs)
estimateSentenceProbabilityLS(word_sequence2, seq2_bigrams, unigrams, bigrams, bigrams_probs, unigrams_probs)
estimateSentenceProbabilityLS(word_sequence3, seq3_bigrams, unigrams, bigrams, bigrams_probs, unigrams_probs)
estimateSentenceProbabilityLS(word_sequence4, seq4_bigrams, unigrams, bigrams, bigrams_probs, unigrams_probs)
estimateSentenceProbabilityLS(word_sequence5, seq5_bigrams, unigrams, bigrams, bigrams_probs, unigrams_probs)



