# DA_nanodegree_project5
## Using machine learning to identify POI in Enron fraud

Data Exploration: 
The goal of this project is to identify person of interest (POI) in the Enron fraud using financial and email data. Machine learning is a powerful tool to discover the underlying relationships in a large volume of dataset and make predictions on new data point. In this dataset, there are a total of 146 data points and 21 features which document the financial information and emails statistics of Enron employees. 18 of them have been identified as POI during the investigation. 128 of data points are not POI. This dataset has a small number of data points and is imbalanced between two classes (POI vs non POI). In the Enron fraud, POIs may have received financial benefit. There may be a high rate of communication between POIs. Therefore, this dataset can be useful for us to learn about predicting POI based on employee’s financial and email data. 
Outlier removal: 
There is one outlier which is the “total” amount of the financial information. This data point is removed since it is not real employee data. Another outlier is CEO, Kenneth Lay. Although Kenneth’s financial information is way out of the general population of the rest of employees, this data point is retained during analysis since it is a real data point.  
Identify NaN:
The dataset does not have complete information for each person’s features. There are many “NaN” values in the dataset. In the following array, the fraction of “NaN” values for each feature is listed. 
[('poi', 0.0),
 ('exercised_stock_option_ratio', 0.0),
 ('poi_email', 0.0),
 ('total_stock_value', 0.137),
 ('total_payments', 0.144),
 ('email_address', 0.233),
 ('restricted_stock', 0.247),
 ('exercised_stock_options', 0.301),
 ('expenses', 0.349),
 ('salary', 0.349),
 ('other', 0.363),
 ('to_messages', 0.404),
 ('shared_receipt_with_poi', 0.404),
 ('from_messages', 0.404),
 ('from_poi_to_this_person', 0.404),
 ('from_this_person_to_poi', 0.404),
 ('bonus', 0.438),
 ('long_term_incentive', 0.548),
 ('deferred_income', 0.664),
 ('deferral_payments', 0.733),
 ('restricted_stock_deferred', 0.877),
 ('director_fees', 0.884),
 ('loan_advances', 0.973)]
Feature engineering:
Other than the original features in the dataset, I engineered two additional features, “poi_email” and “exercised_stock_option_ratio”. “poi_email” is the product of the number of emails from a person to POI and the number of emails from POI to this person. This feature will amplify the effect of a person's email communication with POI during identification. 
“exercised_stock_option_ratio” is the ratio of exercised stock value to the total stock value. POIs are likely to exercise their stock options when they sense the corporation is in trouble. Therefore, the fraction of exercised stock options may be high for POIs. 
Feature selection and model comparison: 
Feature selection is a critical step in machine learning. We want to select as few features as possible to capture the pattern of data and make reliable predictions. I used SelectKBest algorithm to select features. The number of features (k parameter) was determined iteratively using GridSearchCV. The score function for SelectKBest is f1_classif, which determines the correlation between a feature and class label.
I compared three different classifiers and searched the best features for each of them. I created individual pipeline including scaling (only applies to K Nearest Neighbors), feature selection and classification.  
For Naïve Bayes classifier, the pipeline is,
Pipeline(steps=[('filter', SelectKBest(k=7, score_func=<function f_classif at 0x0000000009594AC8>)), ('classifier', GaussianNB())])
The selected best 7 features and their weight are, 
deferred_income 0.217
exercised_stock_option_ratio 6.234
long_term_incentive 11.596
bonus 8.746
total_stock_value 0.03
salary 2.108
exercised_stock_options 24.468
Using the selected features, the performance of the classifier assessed by test_classifier() is,
Pipeline(steps=[('filter', SelectKBest(k=7, score_func=<function f_classif at 0x0000000009594AC8>)), ('classifier', GaussianNB())])
	Accuracy: 0.88800	Precision: 0.71053	Recall: 0.27000	F1: 0.39130	F2: 0.30822
	Total predictions: 1500	True positives:   54	
	False positives:   22	False negatives:  146	
	True negatives: 1278
For K Nearest Neighbors (KNN) Classifier, the pipeline is,
Pipeline(steps=[('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ('filter', SelectKBest(k=4, score_func=<function f_classif at 0x0000000009594AC8>)), ('classifier', KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',metric_params=None, n_jobs=1, n_neighbors=5, p=2, weights='uniform'))])
The selected best 4 features and their weight are, 
exercised_stock_option_ratio 24.2678155059
bonus 21.0600017075
total_stock_value 24.4676540475
exercised_stock_options 25.0975415287
Using the selected features, the performance of the classifier assessed by test_classifier() is,
Pipeline(steps=[('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ('filter', SelectKBest(k=4, score_func=<function f_classif at 0x0000000009594AC8>)), ('classifier', KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=1, n_neighbors=5, p=2, weights='uniform'))])
	Accuracy: 0.88200	Precision: 0.68254	Recall: 0.21500	F1: 0.32700	F2: 0.24913
	Total predictions: 1500	True positives:   43	
	False positives:   20	False negatives:  157	
	True negatives: 1280
For Adaboost Classifier, the pipeline is,
Pipeline(steps=[('filter', SelectKBest(k=15, score_func=<function f_classif at 0x0000000009594AC8>)), ('classifier', AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0, n_estimators=50, random_state=1))])
The selected best 15 features and their weight are, 
expenses 6.23420114051
deferred_income 11.5955476597
exercised_stock_option_ratio 24.2678155059
long_term_incentive 10.0724545294
shared_receipt_with_poi 8.74648553213
loan_advances 7.24273039654
other 4.2049708583
bonus 21.0600017075
total_stock_value 24.4676540475
from_poi_to_this_person 5.34494152315
from_this_person_to_poi 2.42650812724
restricted_stock 9.34670079105
salary 18.575703268
total_payments 8.86672153711
exercised_stock_options 25.0975415287
Using the selected features, the performance of the classifier assessed by test_classifier() is,
Pipeline(steps=[('filter', SelectKBest(k=15, score_func=<function f_classif at 0x0000000009594AC8>)), ('classifier', AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0, n_estimators=50, random_state=1))])
	Accuracy: 0.84267	Precision: 0.39535	Recall: 0.34000	F1: 0.36559	F2: 0.34979
	Total predictions: 1500	True positives:   68	
	False positives:  104	False negatives:  132	
	True negatives: 1196
After testing three classifiers with KBestSelect method, I found that the number of best features selected by KBestSelect algorithm depends on the classifier. Naive Bayer Classifier selected 7 best features. KNN Classifier selected 4 best features. Adaboost Classifier selected 15 best features. I also found that one of the engineered features “exercised_stock_option_ratio” is selected with high weight in all cases, indicating that this feature is very informative in predicting POI. Since these features are evaluated by weight in KBestSelect method, this engineered feature increased model performance. 
With 15 best selected features, Adaboost Classifier achieved relatively good balance of precision (0.395) and recall (0.340). In the next step, further tune Adaboost Algorithm and compare its performance with the simple Naive Bayes algorithm
Given the imbalance of classes and small number of data points, if I used train_test_split, it may cause the training or testing dataset mainly having one class and not able to learn a model to predict another class. Therefore, I have to use “StrattifiedShuffleSplit” as shown in the tester.py to split the data in 100 folds and evaluated the average performance of each classifier on 100 folds. This splitting method gave the same percentage of classes in each training and testing dataset and created 100 folds of shuffled split. This gave more accurate measure of the robustness of classifier. 
Tune an algorithm: 
Tuning an algorithm means to further adjust the hyperparameters of the classifier to achieve better performance. During initial selection of classifier, hyperparameters are chosen by default most of time. By tuning the algorithm, there’s room to improve its performance. I used “grid search cross validation” to find the best choice of the number of estimators and learning rate of the Adaboost Classifier, where I found that 70 base estimators with learning rate of 1.0 gave me best performance. 
parameters = {"n_estimators": [ 60, 65, 70, 75, 80],
              "learning_rate": [0.1, 1., 10.]
             }
sss = StratifiedShuffleSplit(labels,  n_iter=100, random_state = 42)
gs = GridSearchCV(Adaboost, param_grid= parameters, scoring = 'f1', cv =sss )

After tuning, the best algorithm is,

AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
          learning_rate=1.0, n_estimators=70, random_state=1)

Validation: 
Learning a training model from a dataset and accessing its performance on the same dataset can cause problems. This will lead model overfitting and it won’t be able to perform well on “new” or “unseen” dataset. To avoid overfitting, we have to split the dataset into training and testing dataset. However, during tuning an algorithm, if the model is assessed with the testing dataset over and over again, the model will tend to over fit the test dataset and lead to overfitting. In order to avoid this problem, we have to set aside another dataset called, validation dataset. This validation set is only used to evaluate the learned model during tuning. Its final performance is judged by the completely new testing dataset. This will avoid the overfitting problem.
In this project, I used stratified shuffle cross validation during algorithm tuning. Give the same number of data points and heavy imbalance of classes, stratified shuffle cross validation can split shuffled data into stratified training and validation data sets in 100 folds. For each fold, the model was fitted into the training set and the performance was evaluated on the validation data set. The average performance of the classifier on 100 folds was reported to provide more reliable evaluation.
Evaluation: 
I used several evaluation metrics to evaluate the classifier’s performance, including accuracy, precision, recall and F-1 score. For my final tuned model, the accuracy is 0.851 and F1 score is 0.375. Accuracy reflect the percentage of prediction that matches the true label of the data points, which means 85.4% chance of prediction of POI or non POI is correct. F1 score reflects the balance of precision (0.426) and recall (0.335). Precision tells us that 43.2% of predicted POI is correct. Recall tells us that a real POI has 30.5% chance to be correctly predicted. Since we want to capture all potential POIs without wronging innocent people, we have to find a balance of precision and recall, which is reflected by F1 score.  
