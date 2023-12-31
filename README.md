﻿# Master's work prototype

## DETECTING MESSAGES WITH THREATENING CONTENT IN LITHUANIAN TEXT

---

The link to the project prototype [can be found here](https://threats-in-lt-detection.onrender.com). The server on the request first has to spin up, so be patient to wait.

---

Every test, training and prediction with every model can checked in fullproject.ipynb

## The annotation

The aim o f the Master's thesis is to apply machine learning algorithms to recognize text with threatening content. To achieve the goal, the
following tasks were set: to perform an analysis o f the methods used for text recognition, to collect a data set that will be used t o train the
aforementioned algorithms, to propose a method o r a combination o f methods that would help to implement the work's purpose, and to implement the
proposed methods and present the results. The novelty o f the topic is that such a study has not yet been carried out for the Lithuanian language.
Relevance of the topic - contributed to the study of the Lithuanian language in natural language processing. A realized goal is also relevant as a filter
that filters out threatening sentences. After analyzing the methods, it was found that there are many classifiers for text classification, which are divided
into classic and deep learning algorithms. Each classifier has advantages and disadvantages, so the model should be chosen according to the existing
problem and the desired goal. The data set was collected from various media portals and sources existing in other languages, which were translated
into Lithuanian. The set consisted of 500 threatening and 500 neutral sentences. For research, it was proposed to use TF and TF-IDF methods, to try
dimensionality reduction and to test different parameters of classifiers. For the final test, Naive Bayess, Support Vector Machine, BERT, Decision Trees, Gradinet Boosted, Random
Forest and Multi Layer Perceptron classifiers were chosen, T F and T F-IDF techniques and LSA dimensionality reduction were tested. The results showed that dimensionality reduction improved
only DT and GB classifiers overall accuracy, all other classifiers dropped. The TF-IDF method was superior in all cases except for the NB classifier,
where the TF method had a small advantage. The best accuracies were NB at 90%, SVM at 89%, and MLP at 86%. The most threatening sentences
were guessed by the NB classifier, second to MLP, so these classifiers were chosen to be integrated into the sentence classification API.

---

## Technical part

To achieve the goals of this project "python" scripting language was chosen for it's wide selection of AI and machine learning libraries.  
The classifiers's results were scored based on 4 calculations: accuracy, sensitivity, F-score and overall accuracy.

### Text classification process

- Collect text documents
- Text processing
- Feature extraction
- Classification
- Model assessment

---

### Process Explanation

**Documents (sentences)** were collected by hand for a few reasons:

1. All webscraping tools require extra effort to make them work properly (VPN, spam blocking), which was not part of the project.
2. Even with web scraping, data would be needed to checked and marked correctly, which would take that extra time.  
   Every sentence was classified by my self with either 0 - not threatening or 1 - threatening.
   
**Text proccessing** was done in these steps: letter conversion to lowercase, filtering all punctuation marks and other unneccessary signs, filtering short words, extra spaces filtering, number filtering, stop words filtering, tokenization, stemming. This was done to reduce the amount of noise in the text, reduce the amount of data to train, reduce irrelevant information which does not add importat meaning.  
**Feature extraction** was done with Term Frequency and Term Frequency - Inverse Document Frequency. This was done so that the models would understand what is given to them.  
**Classification** was done by training, evaluating and predicting with the models.  
**Model assesment** was done by 4 calculations with different parameters.
