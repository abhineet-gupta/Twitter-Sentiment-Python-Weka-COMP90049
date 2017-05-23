# Twitter Sentiment Analysis

## COMP90049 Knowledge Technologies

University of Melbourne

Assignment 2 - Sem 1, 2017

---

### Report

- rationale behind the feature engineering
- systems used i.e. the ML algorithms
- performance over dev data and observations
- "knowledge"
- **DON'T** include name or ID

### Readme

- describe how to generate the features
- purposes of any important scripts or resources
- various details which are outputs on the accompanying test data
  - which models and parameters used to generate it
- system settings, resource limits, etc.
- **NOT** required to submit scripts that generate output of the system

### Code

- transformation of data into a format to be used in the ML system e.g. ARFF for Weka
  - read tweets; lower case, remove non-alphabets
  - remove stop words
  - calc word freq
  - calc gini idx
  - select words where gini < x and freq > y
  - sort on gini [asc]
  - calc custom idx
  - limit features
  - generate .arff for test data
  - run in Weka
    - NB
    - DT
    - KNN

- Feature engineering OTHER than frequency of words for full marks e.g. bi-grams

### Deliverables

- MSE
  - code
  - Readme
  - Output of test results

- LMS
  - report
  - reviews
