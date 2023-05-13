Georgia Tech Data Science and Analytics BootCamp - May 2023

Homework Module 20 - Supervised Machine Learning - Credit Risk Classification Challenge
By Priscila Menezes Briggs


# Credit Risk Classification Report

## Overview of the Analysis

In this analysis, various techniques were used to train and evaluate a model based on loan risk. A dataset of historical lending activity from a peer-to-peer lending services company  was used to build a model that can identify the creditworthiness of borrowers.

The purpose of this analysis is making the model identify which types of loans are considered healthy and which of them should be classified as high-risk loans. 

Some of the variables that are being predicted by the models, before and after resampling the data, are:

* balance of target values, using value_counts function;
* balanced accuracy score;
* precision, recall and F-1 scores.

The stages of this analysis are divided into the following subsections:

* Split the Data into Training and Testing Sets
  * separate label column (loan_status) from remaining columns
  * scale the data

* Create a Logistic Regression Model with the Original Data
  * instantiate the logistic regression model with random_state parameter of 1
  * fit the model using training data
  * make predictions using the testing data
  * evaluate the model's performance

* Predict a Logistic Regression Model with Resampled Training Data
  * instantiate the random oversampler model with random_state parameter of 1
  * Fit the original training data to the random_oversampler model
  * split the resampled data into training and testing sets
  * scale the resampled data
   * instantiate the logistic regression model using resampled data with random_state parameter of 1
  * fit the model using resampled training data
  * make predictions using the resampled testing data
  * evaluate the model's performance

The main method used to develop the models was the Logistic Regression, first applied to the original scaled data, then later using the Random OverSampler model to resample the data and calculate new results with the Logistic Regression model. The accuracy of the models was calculated with the balanced accuracy score, as this is the preferred method used when dealing with imbalanced data, which is the case with the resource data set.


## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
   
  * Accuracy: The balanced accuracy score for this model was 0.9889. This is the best measure for this model, instead of accuracy, as it is used when dealing with imbalanced data, that is, when one of the target classes appears more than the other, which is the case of this data set, as shown in the support numbers (the number of occurrences of healthy loans is much higher than the occurrences of high-risk loans).
  * Precision: This model is very accurate in evaluating true positives against healthy loans (it correctly identifies healthy loans, with precision equals to 1.00). However, it could do a better job of evaluating high-risk occurrences, as the precision for this case is only 0.84. This means that the model still does not flag high-risk real loans so precisely, having flagged some loans that were actually healthy as high risk (= higher value of 'false positives'). 
  * Recall: The model makes very few mistakes in identifying negative cases, both for healthy loans and for high-risk loans, with minimal difference between the performance of each one of them (0.99 and 0.98 respectively for healthy and high-risk loans).

* Machine Learning Model 2 (resampled data):
  * Accuracy: 0.9948.
  * Precision: 1.00 and 0.99 respectively for healthy and high-risk loans.
  * Recall: 0.99 and 1.00 respectively for healthy and high-risk loans.
  
  * This model outperforms the previous model as demonstrated by increased precision, recall, and F-1 score for high-risk loans, having the precision presented the biggest increase. The model kept the same values as the previous one for healthy loans, but the balanced accuracy score also got higher, becoming 0.9948. This shows that oversampling the least represented data (high-risk loans) so that the machine has more opportunities to analyze these types of loans led to a better model response in identifying each one of them.

## Summary

This analysis shows that the logistic regression model with resampled data presented better results than the model that used the original data as a resource, so the model with resampled data (model #2) is the most recommended for this case.
Model #2 performs best on all measures relating to high-risk loans. The use of this model is highly recommended, as it is essential that all financial institutions be able to identify high-risk loans, as these can cause great losses in the institution's accounting.

## Availability to the public
The files used in this challenge are available in the GitHub's repository on https://github.com/PrisBriggs/credit-risk-classification .


## References:

The references used in this Challenge were the activities and lessons given in class, the tutoring classes, and the websites below. 

All webpages were visited in May/2023.

https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.RandomOverSampler.html
https://stackoverflow.com/questions/50376990/modulenotfounderror-no-module-named-imblearn
https://medium.com/@kohlishivam5522/understanding-a-classification-report-for-your-machine-learning-model-88815e2ce397
https://neptune.ai/blog/balanced-accuracy
https://developers.google.com/machine-learning
https://towardsdatascience.com/what-and-why-behind-fit-transform-vs-transform-in-scikit-learn-78f915cf96fe