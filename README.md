# Twitter Sentiment Analysis

## COMP90049 Knowledge Technologies

University of Melbourne

Project 2 - Sem 1, 2017

### Project specification can be found [here](2017S1-KT-proj2-spec.pdf)

---

Run ```freq.py```, assuming the data is present in the folder structure supplied.

Output is present in ```data/weka-output```

Use Weka to analyse the output [pre-processed tweets] via machine learning algorithms.

---

### Brief summary of code

The program transforms data into a format to be used in the ML system e.g. ARFF for Weka. It does the following in order:

- read tweets; lower case, remove non-alphabets
- remove stop words
- calculate word frequencies
- calculate GINI index
- select words where GINI < x and freq > y
- sort on GINI [asc]
- calc custom index
- limit features
- generate .arff for test data
- run in Weka with following ML algorithms
  - Naive Bayes
  - Decision trees
  - K-Nearest Neighbours