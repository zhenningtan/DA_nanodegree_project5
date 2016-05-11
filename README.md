# DA_nanodegree_project5
## Using machine learning to identify POI in Enron fraud

*1. Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]* 

The goal of this project is to idenfity person of interest (POI) in the Enron fraud using financial and email statistics data. Machine learning is a powerful tool to find the underlying relationships in a large number of dataset and make predictions on new dataset to help us idenfity potential POI. In this dataset, there are a total of 146 data points and 21 features which document the finaical information and emails statistics of Enron employees. 18 of them have been identified as POI during the investigation. 128 of data points are not POI. Due to the small number of data points and imbalance between two classes (POI vs non POI), it is important to build simple and robust, unbiased machine learning algorithm to make predictions. 

number of POI: 18 
number of non-POI: 128
number of features: 21
Most features have missing value except “poi” feature. I will only consider to use features with missing value in less than 0.5
One outlier is the “total” data point which sums the amount of all the entries. This outlier is removed 

What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “properly scale features”, “intelligently select feature”]
What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]
What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric item: “tune the algorithm”]
What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric item: “validation strategy”]
Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]
