
# ROC Curves and AUC - Lab


## Introduction 

In this lab, you'll practice drawing ROC graphs, calculating AUC, and interpreting these results. In doing so, you will also further review logistic regression, by briefly fitting a model as in a standard data science pipeline.

## Objectives

You will be able to:

* Evaluate classification models using various metrics
* Define and understand ROC and AUC

## Training the Model

Start by repeating the previous modeling steps we have discussed. For this problem, you are given a dataset **mushrooms.csv**. Your first job is to train a LogisticRegression classifier on the dataset to determine whether the mushroom is **e**dible or **p**oisonous. The first column of the dataset *class* indicates whether or not the mushroom is poisonous or edible.

** For consistency use random_state=0**


```python
#Your code here
```


```python
# __SOLUTION__ 
#Your code here
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd


#Load the data
df = pd.read_csv('mushrooms.csv')

#Data Preview
df.head()
```


```python
# __SOLUTION__ 
#Further data previews
df.info()
```


```python
# __SOLUTION__ 
#Define appropriate X and y
X = df[df.columns[1:]]
y = pd.get_dummies(df["class"], drop_first=True)

#Create Dummy Variables
X = pd.get_dummies(X, drop_first=True)
# Split the data into train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

#Fit a model
logreg = LogisticRegression(fit_intercept = False, C = 1e12, solver='liblinear') #Starter code
model_log = logreg.fit(X_train, y_train.values.ravel())
print(model_log) #Preview model params

#Predict
y_hat_test = logreg.predict(X_test)
```

## ROC Metrics
  
Next, calculate the false positive rate and true positive rate (you can use the built-in metrics from sci-kit learn) of your classifier.


```python
# Your code here
```


```python
# __SOLUTION__ 
# Your code here
from sklearn.metrics import roc_curve, auc

#for various decision boundaries given the case member probabilites

#First calculate the probability scores of each of the datapoints:
y_score = model_log.decision_function(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_score)

y_train_score = model_log.decision_function(X_train)
train_fpr, train_tpr, thresholds = roc_curve(y_train, y_train_score)
```

## Drawing the ROC Graph

Next, use the false positive rate and true positive rate to plot the Receiver Operating Characteristic Curve for both the train and test sets.


```python
# Your code here
```


```python
# __SOLUTION__ 
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#Seaborns Beautiful Styling
sns.set_style("darkgrid", {"axes.facecolor": ".9"})

plt.figure(figsize=(10,8))
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.yticks([i/20.0 for i in range(21)])
plt.xticks([i/20.0 for i in range(21)])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC) Curve for Test Set')
plt.legend(loc="lower right")
print('AUC: {}'.format(auc(fpr, tpr)))
plt.show()
```


```python
# __SOLUTION__ 
#Seaborns Beautiful Styling
sns.set_style("darkgrid", {"axes.facecolor": ".9"})

plt.figure(figsize=(10,8))
lw = 2
plt.plot(train_fpr, train_tpr, color='darkorange',
         lw=lw, label='ROC curve')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.yticks([i/20.0 for i in range(21)])
plt.xticks([i/20.0 for i in range(21)])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC) Curve for Training Set')
plt.legend(loc="lower right")
print('AUC: {}'.format(auc(train_fpr, train_tpr)))
plt.show()
```

## Interpretation:

What do you notice about these ROC curves?

## Your answer here


```python
# __SOLUTION__
# Both have an AUC of 1.0, indicating their performance is perfect.
# Note that this is an extreme rarity! 
# Typically, if models perform this well it is too good to be true.
```

## Interpretation
Look at the ROC curve graph from the lesson:  

<img src="images/lesson_roc_graph.png">

Think about the scenario of this model: predicting heart disease. If you tune the current model to have an 82% True Positive Rate, (you've still missed 20% of those with heart disease), what is the False positive rate?


```python
fpr = #write the approximate fpr when tpr=.8
```


```python
# __SOLUTION__ 
fpr = .17 #write the approximate fpr when tpr=.8
```

## Interpretation 2
If you instead tune the model to have a 95.2% True Postive Rate, what will the False Postive Rate be?


```python
fpr = #write the approximate fpr when tpr=.95
```


```python
# __SOLUTION__ 
fpr = .22 #write the approximate fpr when tpr=.95
```

## Opinion
In the case of heart disease dataset that we've been talking about, do you find any of the above cases acceptable? How would you tune the model? Describe what this would mean in terms of the number of patients falsely scared of having heart disease and the risk of missing the warning signs for those who do actually have heart disease.

## Your answer here


```python
# __SOLUTION__

# With such an important decision, such as detecting heart disease, we would hope for more accurate results. 
# The True positive weight is the more important of the two in this scenario. 
# That is, the true positive rate determines the percentage of patients with heart disease who are correctly identified and warned. 
# The false positive rate is still very important, but it would be better to accidentally scare a few healthy patients 
# and warn them of potentially having heart disease than having missed warnings. 
# That said, the false positive rate becomes rather unacceptably high once the true positive rate exceeds .95. 
# A .95 TPR indicates that out of 100 patients with heart disease we correctly warn 95 of them, but fail to warn 5. 
# At the same time, this has a FPR of nearly .25 meaning that roughly one in four times we incorrectly warn a patient of heart disease 
# when they are actually healthy.
```

## Summary

In this lab you further explored ROC curves and AUC, drawing graphs and then interpreting these results to lead to a more detailed and contextualized understanding of your model's accuracy.
