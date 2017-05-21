#!/usr/bin/python3
"""
COMP90049 Sem 1, 2017
Assignment 2
Twitter Sentiment analysis
"""
import re
import os

FILE_PATH_SMALL = os.path.join(os.path.dirname(__file__), "../data/small-test.txt")
FILE_PATH_TRAIN = os.path.join(os.path.dirname(__file__), "../data/orig/train-tweets.txt")
OUT_FILE_PATH = os.path.join(os.path.dirname(__file__), "../data/output/freq.csv")

RE_CLEAN_TWEET = r'(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)'

def import_tweets(path):
    '''
    imports tweets from file into a dict
    ...and cleans up the text
    @return dict containing IDs as key and tweets as value
    '''
    tweet_file = open(path, encoding="utf8")
    tweets = {}
    for line in tweet_file:
        temp = line.strip().split("\t")
        tweets[temp[0]] = ' '.join(re.sub(RE_CLEAN_TWEET, " ", temp[1]).split()).lower()
    
    tweet_file.close()
    return tweets

def calc_word_freq(tweets):
    '''
    counts frequency of each word appearing in every tweet
    @param tweets a dict of {ID: tweet} values
    '''
    _words_freq = {}
    for tid in tweets:
        temp_tweet = tweets[tid].split(" ")
        for word in temp_tweet:
            if word in _words_freq:
                _words_freq[word] += 1
            else:
                _words_freq[word] = 1
    return _words_freq

def main():
    '''Main function'''
    # contains tweets in a dict with ID as key and tweet as value
    tweets = import_tweets(FILE_PATH_TRAIN)
    print("Tweets read and collected:", len(tweets))

    # Calculate word frequencies for all words
    words_freq = calc_word_freq(tweets)
    print(len(words_freq), "words counted")

    # Write out results to CSV
    csv_out = open(OUT_FILE_PATH, 'w')
    for key in words_freq:
        csv_out.write(key + "," + str(words_freq[key]) + "\n")
    csv_out.close()

if __name__ == '__main__':
    main()
