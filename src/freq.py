#!/usr/bin/python3
"""
COMP90049 Sem 1, 2017
Assignment 2
Twitter Sentiment analysis
"""
import re
import os

STOP_WORD_LEN = 3
GINI_CUTOFF = 0.55
FEATURE_SIZE = 100
FREQ_CUTOFF = 30

PREPEND_FP = os.path.dirname(__file__)

FP_TRAIN_TWEETS_S = os.path.join(PREPEND_FP, "../data/small-train-tweets.txt")
FP_TRAIN_LABELS_S = os.path.join(PREPEND_FP, "../data/small-train-labels.txt")

FP_TRAIN_TWEETS = os.path.join(PREPEND_FP, "../data/orig/train-tweets.txt")
FP_TRAIN_LABELS = os.path.join(PREPEND_FP, "../data/orig/train-labels.txt")
FP_DEV_TWEETS = os.path.join(PREPEND_FP, "../data/orig/dev-tweets.txt")
FP_DEV_LABELS = os.path.join(PREPEND_FP, "../data/orig/dev-labels.txt")
FP_TEST_TWEETS = os.path.join(PREPEND_FP, "../data/orig/test-tweets.txt")

OUT_FILE_PATH = os.path.join(PREPEND_FP, "../data/output/freq.csv")

RE_CLEAN_TWEET = r'(@[A-Za-z0-9]+)|([^A-Za-z \t])|(\w+:\/\/\S+)'

def import_tweets(path):
    '''
    imports tweets from file into a dict
    ...and cleans up the text
    @param path filepath for tweets file containing ID and tweets
    @return dict containing IDs as key and tweets as value
    '''
    tweet_file = open(path, encoding="utf8")
    tweets = {}
    for line in tweet_file:
        temp = line.strip().split("\t")
        tweets[temp[0]] = ' '.join(re.sub(RE_CLEAN_TWEET, " ", temp[1]).split()).lower()

    tweet_file.close()
    return tweets

def import_labels(path):
    '''
    imports labels for each tweet ID
    @param path filepath for labels with id and label, tab separated
    @return dict containing tweet ID and its label
    '''
    label_file = open(path)
    labels = {}
    for line in label_file:
        temp = line.strip().split("\t")
        labels[temp[0]] = temp[1]
    return labels

def calc_word_freq(tweets, labels):
    '''
    counts frequency of each word appearing in every tweet
    @param tweets a dict of {ID: tweet} values
    @param labels a dict of {ID: label}
    @return dict containing {word: [pos, neu, neg]}
    '''
    _words_freq = {}
    for tid in tweets:
        temp_tweet = tweets[tid].split(" ")
        for word in temp_tweet:
            if word in _words_freq:
                if labels[tid] == 'positive':
                    _words_freq[word][0] += 1
                elif labels[tid] == 'neutral':
                    _words_freq[word][1] += 1
                else:   # negative
                    _words_freq[word][2] += 1
            else:
                if labels[tid] == 'positive':
                    _words_freq[word] = [1, 0, 0]
                elif labels[tid] == 'neutral':
                    _words_freq[word] = [0, 1, 0]
                else: # negative
                    _words_freq[word] = [0, 0, 1]
    return _words_freq

def sum_word_freq(word_freq):
    '''
    calculates sum of all categories of word freq
    @param word_freq dict containing {word: [pos, neu, neg]}
    @return dict containing {word: total}
    '''
    word_freq_total = {}
    for word in word_freq:
        word_freq_total[word] = sum(word_freq[word])
    return word_freq_total

def remove_stop_words(word_freq, word_len):
    '''
    removes words of length 1 or 2 characters
    @param dict of {word: [pos, neu, neg]}
    @return dict same as param but without stop words
    '''
    result = {}
    for word in word_freq:
        if len(word) > word_len:
            result[word] = word_freq[word]
    return result

def calc_gini(words_freq, words_freq_sum):
    '''
    Calculates gini index for each word
    @param dict {word: [pos, neu, neg]} and dict {word: total_freq}
    @return dict {word: gini}
    '''
    result = {}

    for word in words_freq:
        freq = words_freq_sum[word]
        result[word] = 1-((words_freq[word][0]/freq)**2 + \
            (words_freq[word][1]/freq)**2 + (words_freq[word][2]/freq)**2)
    return result

def filter_gini_freq(words_gini, words_freq_sum, gini_cutoff, freq_cutoff):
    '''
    Filter words based only on their gini idx and total freq
    @param words_gini dict{word: gini_idx}
    @param words_freq_sum dict{word: sum_freq}
    @param gini_cutoff value between 0 and 1 used to filter
    @param freq_cutoff frequency to filter by
    @return dict {word: gini} that passed criterion
    '''
    results = {}
    for word in words_gini:
        if words_gini[word] < gini_cutoff:
            if words_freq_sum[word] > freq_cutoff:
                results[word] = words_gini[word]

    return results

def main():
    '''Main function'''
    # contains tweets in a dict with ID as key and tweet as value
    tweets = import_tweets(FP_TRAIN_TWEETS)
    print("Tweets read and collected:", len(tweets))

    # import labels in a dict with ID as key and label as value
    labels = import_labels(FP_TRAIN_LABELS)
    print("Labels read:", len(labels))

    # Calculate word frequencies for all words
    raw_words_freq = calc_word_freq(tweets, labels)
    print(len(raw_words_freq), "words counted")

    # remove stop words
    words_freq = remove_stop_words(raw_words_freq, STOP_WORD_LEN)
    print(str(len(raw_words_freq) - len(words_freq)), "stop words removed.")

    # sum word freq
    words_freq_sum = sum_word_freq(words_freq)

    # calc gini index for each word
    words_gini = calc_gini(words_freq, words_freq_sum)

    # Filter words based on freq sum and gini
    words_gini_filtered = filter_gini_freq(words_gini, words_freq_sum, GINI_CUTOFF, FREQ_CUTOFF)

    # # sort based on word_freqs
    # sorted_wf_sum_list = [(k, words_freq_sum[k]) for k in sorted(
    #     words_freq_sum, key=words_freq_sum.get, reverse=True)]

    # sort based on gini
    sorted_w_gini_list = [(k, words_gini_filtered[k]) for k in sorted(
        words_gini_filtered, key=words_gini_filtered.get, reverse=False)]

    # Write out results to CSV
    csv_out = open(OUT_FILE_PATH, 'w')
    csv_out.write("word" + "," + \
        "freq_positive" + "," + "freq_neutral" + "," + "freq_negative" + \
        "," + "total_freq" + "," + "gini_idx" + "\n")

    for word, gini in sorted_w_gini_list:
        csv_out.write(word + "," + str(words_freq[word][0]) + \
            "," + str(words_freq[word][1]) + \
            "," + str(words_freq[word][2]) + \
            "," + str(words_freq_sum[word]) + \
            "," + str(gini) + \
            "\n")   # Calculate GINI index for each attribute
    csv_out.close()

if __name__ == '__main__':
    main()
