#!/usr/bin/python3
"""
COMP90049 Sem 1, 2017
Assignment 2
Twitter Sentiment analysis
"""
import re
import os

FP_TRAIN_TWEETS_S = os.path.join(os.path.dirname(__file__), "../data/small-train-tweets.txt")
FP_TRAIN_LABELS_S = os.path.join(os.path.dirname(__file__), "../data/small-train-labels.txt")

FP_TRAIN_TWEETS = os.path.join(os.path.dirname(__file__), "../data/orig/train-tweets.txt")
FP_TRAIN_LABELS = os.path.join(os.path.dirname(__file__), "../data/orig/train-labels.txt")
FP_DEV_TWEETS = os.path.join(os.path.dirname(__file__), "../data/orig/dev-tweets.txt")
FP_DEV_LABELS = os.path.join(os.path.dirname(__file__), "../data/orig/dev-labels.txt")
FP_TEST_TWEETS = os.path.join(os.path.dirname(__file__), "../data/orig/test-tweets.txt")

OUT_FILE_PATH = os.path.join(os.path.dirname(__file__), "../data/output/freq.csv")

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
    @return dict containing {word: [pos, neu, neg, tot]}
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

def main():
    '''Main function'''
    # contains tweets in a dict with ID as key and tweet as value
    tweets = import_tweets(FP_TRAIN_TWEETS)
    print("Tweets read and collected:", len(tweets))

    # import labels in a dict with ID as key and label as value
    labels = import_labels(FP_TRAIN_LABELS)
    print("Labels read:", len(labels))

    # Calculate word frequencies for all words
    words_freq = calc_word_freq(tweets, labels)
    print(len(words_freq), "words counted")

    # sum word freq
    words_freq_sum = sum_word_freq(words_freq)

    # sort word_freqs
    sorted_wf_sum_list = [(k, words_freq_sum[k]) for k in sorted(
        words_freq_sum, key=words_freq_sum.get, reverse=True)]

    # Write out results to CSV
    csv_out = open(OUT_FILE_PATH, 'w')
    csv_out.write("tweet_ID" + "," + \
        "freq_positive" + "," + "freq_neutral" + "," + "freq_negative" + \
        "," + "total_freq" + "\n")

    for word, freq in sorted_wf_sum_list:
        csv_out.write(word + "," + str(words_freq[word][0]) + \
            "," + str(words_freq[word][1]) + \
            "," + str(words_freq[word][2]) + \
            "," + str(freq) + "\n")
    csv_out.close()

if __name__ == '__main__':
    main()
