#!/usr/bin/python3
"""
COMP90049 Sem 1, 2017
Assignment 2
Twitter Sentiment analysis
"""
import re
import os

STOP_WORD_LEN = 3
GINI_ZERO_REPLACE = 0.1

GINI_CUTOFF = 0.50
FREQ_CUTOFF = 20
FEATURE_SIZE = 50

PREPEND_FP = os.path.dirname(__file__)

FP_TRAIN_TWEETS_S = os.path.join(PREPEND_FP, "data/small-train-tweets.txt")
FP_TRAIN_LABELS_S = os.path.join(PREPEND_FP, "data/small-train-labels.txt")

FP_TRAIN_TWEETS = os.path.join(PREPEND_FP, "data/orig/train-tweets.txt")
FP_TRAIN_LABELS = os.path.join(PREPEND_FP, "data/orig/train-labels.txt")
FP_DEV_TWEETS = os.path.join(PREPEND_FP, "data/orig/dev-tweets.txt")
FP_DEV_LABELS = os.path.join(PREPEND_FP, "data/orig/dev-labels.txt")
FP_TEST_TWEETS = os.path.join(PREPEND_FP, "data/orig/test-tweets.txt")

OUT_FILE_PATH = os.path.join(PREPEND_FP, "data/freq.csv")
FP_OUT_TRAIN1_ARFF = os.path.join(PREPEND_FP, "data/weka-input/train-custom.arff")
FP_OUT_DEV1_ARFF = os.path.join(PREPEND_FP, "data/weka-input/dev-custom.arff")
FP_OUT_TEST1_ARFF = os.path.join(PREPEND_FP, "data/weka-input/test-custom.arff")

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
    label_file.close()
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
                elif labels[tid] == 'negative':
                    _words_freq[word][1] += 1
                else:   # neutral
                    _words_freq[word][2] += 1
            else:
                if labels[tid] == 'positive':
                    _words_freq[word] = [1, 0, 0]
                elif labels[tid] == 'negative':
                    _words_freq[word] = [0, 1, 0]
                else: # neutral
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

def calc_custom_idx(words_freq, words_gini_filtered):
    '''
    calculates product of freq * 1/gini for each word
    @param word freq dict and word gini dict
    @return word custom idx dict
    '''
    result = {}
    for word in words_gini_filtered:
        temp_gini = words_gini_filtered[word]
        if temp_gini == 0:
            temp_gini = GINI_ZERO_REPLACE
        result[word] = words_freq[word] * 1/temp_gini
    return result

def sort_dict_on_values(dic, desc=False):
    '''
    sorts a dictionary on its values
    @param: dic dictionary {key: value}
    @param: desc True if sorting descending
    @return: a list of (key,value) tuples sorted by value
    '''
    return [(k, dic[k]) for k in sorted(dic, key=dic.get, reverse=desc)]

def gen_csv_features(s_list, words_freq, words_freq_sum, words_gini, file_path):
    '''
    generate a csv file containing selected features
    @param: list of features sorted by some metric about them
    @param: frequency of words in each class
    @param: total freq of words
    @param: gini index of words
    @param: file path to output to
    '''
    csv_out = open(file_path, 'w')
    csv_out.write("word" + "," + \
        "freq_positive" + "," + "freq_negative" + "," + "freq_neutral" + \
        "," + "total_freq" + "," + "gini_idx" + "," + "custom_idx" + "\n")

    for word, cidx in s_list[:FEATURE_SIZE]:
        csv_out.write(word + "," + str(words_freq[word][0]) + \
            "," + str(words_freq[word][1]) + \
            "," + str(words_freq[word][2]) + \
            "," + str(words_freq_sum[word]) + \
            "," + str(words_gini[word]) + \
            "," + str(cidx) + \
            "\n")
    csv_out.close()

def gen_arff(feature_list, instances, labels, out_file_path):
    '''
    generates .arff file for use with Weka
    @param: list of features
    @param: instances those features will either contain or not; dict {id: tweet word list}
    @param: labels for each instance
    @param: file path for .arff file
    '''
    f_contents = ""
    f_contents += "@RELATION twitter-sent-top20\n"
    for feature in feature_list:
        f_contents += "@ATTRIBUTE " + feature + " NUMERIC\n"
    f_contents += "@ATTRIBUTE sentiment {positive,negative,neutral}\n"
    f_contents += "@DATA\n"

    for instance in instances:
        temp_tweet = instances[instance].split(" ")
        # f_contents += instance + ","
        for feature in feature_list:
            if feature in temp_tweet:
                f_contents += str(temp_tweet.count(feature)) + ","
            else:
                f_contents += "0,"
        f_contents += labels[instance]
        f_contents += "\n"
    arff = open(out_file_path, 'w')
    arff.write(f_contents)
    arff.close()

def main():
    '''Main function'''
    # contains tweets in a dict with ID as key and tweet as value
    tweets_train = import_tweets(FP_TRAIN_TWEETS)
    tweets_dev = import_tweets(FP_DEV_TWEETS)
    tweets_test = import_tweets(FP_TEST_TWEETS)
    print("Tweets from training read:", len(tweets_train))
    print("Tweets from dev read:", len(tweets_dev))
    print("Tweets from test read:", len(tweets_test))

    # import labels in a dict with ID as key and label as value
    labels_train = import_labels(FP_TRAIN_LABELS)
    labels_dev = import_labels(FP_DEV_LABELS)
    print("Labels from train read:", len(labels_train))
    print("Labels from dev read:", len(labels_dev))
    # label for test data is a question mark
    labels_test = {}
    for tweet_id in tweets_test:
        labels_test[tweet_id] = "?"

    # Calculate word frequencies for all words
    raw_words_freq = calc_word_freq(tweets_train, labels_train)
    print(len(raw_words_freq), "words counted")

    # remove short words
    words_freq = remove_stop_words(raw_words_freq, STOP_WORD_LEN)
    print(str(len(raw_words_freq) - len(words_freq)), "stop words removed.")

    # sum word freq
    words_freq_sum = sum_word_freq(words_freq)

    # calc gini index for each word
    words_gini = calc_gini(words_freq, words_freq_sum)

    # Filter words based on freq sum and gini
    words_gini_filtered = filter_gini_freq(words_gini, words_freq_sum, GINI_CUTOFF, FREQ_CUTOFF)

    # calc custom IDX
    words_custom_idx = calc_custom_idx(words_freq_sum, words_gini_filtered)

    # sort based on a metric
    sorted_w_custom_list = sort_dict_on_values(words_custom_idx, True)

    # Write out results to CSV
    gen_csv_features(sorted_w_custom_list, words_freq, words_freq_sum, words_gini, OUT_FILE_PATH)

    feat_list = []
    for feature in sorted_w_custom_list[:FEATURE_SIZE]:
        feat_list.append(feature[0])

    # write out arff files
    gen_arff(feat_list, tweets_train, labels_train, FP_OUT_TRAIN1_ARFF)
    gen_arff(feat_list, tweets_dev, labels_dev, FP_OUT_DEV1_ARFF)
    gen_arff(feat_list, tweets_test, labels_test, FP_OUT_TEST1_ARFF)

if __name__ == '__main__':
    main()
