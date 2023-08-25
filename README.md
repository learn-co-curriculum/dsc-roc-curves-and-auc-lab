# ROC Curves and AUC - Lab


## Introduction 

In this lab, you'll practice drawing ROC graphs, calculating AUC, and interpreting these results. In doing so, you will also further review logistic regression, by briefly fitting a model as in a standard data science pipeline.

## Objectives

You will be able to:

- Create a visualization of ROC curves and use it to assess a model 
- Evaluate classification models using the evaluation metrics appropriate for a specific problem 

## Train the model

Start by repeating the previous modeling steps we have discussed. For this problem, you are given a dataset `'mushrooms.csv'`. Your first job is to train a `LogisticRegression` classifier on the dataset to determine whether the mushroom is edible (e) or poisonous (p). The first column of the dataset `class` indicates whether or not the mushroom is poisonous or edible.

But first, 

- Import the data 
- Print the first five rows of the data 
- Print DataFrame's `.info()` 


```python
# Import and preview the data


df = None



```

The next step is to define the predictor and target variables. Did you notice all the columns are of type `object`? So you will need to first create dummy variables. 

- First, create a dummy variable for the `'class'` column. Make sure you drop the first level 
- Drop the `'class'` column from `df` and then create dummy variables for all the remaining columns. Again, make sure you drop the first level 
- Import `train_test_split` 
- Split the data (`X` and `y`) into training and test sets with 25% in the test set. Set `random_state` to 42 to ensure reproducibility 


```python
# Define y
y = None
y = y['p']

# Define X
X = None
X = None

# Import train_test_split


# Split the data into training and test sets
X_train, X_test, y_train, y_test = None
```

- Fit the vanilla logistic regression model we defined for you to training data 
- Make predictions using this model on test data 


```python
# Import LogisticRegression
from sklearn.linear_model import LogisticRegression

# Instantiate
logreg = LogisticRegression(fit_intercept=False, C=1e12, solver='liblinear')

# Fit the model to training data
model_log = None

# Predict on test set
y_hat_test = None
```

## Calculate TPR and FPR
  
Next, calculate the false positive rate and true positive rate (you can use the built-in functions from `sklearn`): 


```python
# Import roc_curve, auc


# Calculate the probability scores of each point in the training set
y_train_score = None

# Calculate the fpr, tpr, and thresholds for the training set
train_fpr, train_tpr, thresholds = None

# Calculate the probability scores of each point in the test set
y_score = None

# Calculate the fpr, tpr, and thresholds for the test set
fpr, tpr, thresholds = None
```

## Draw the ROC curve

Next, use the false positive rate and true positive rate to plot the Receiver Operating Characteristic Curve for both the train and test sets.


```python
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Seaborn's beautiful styling
sns.set_style('darkgrid', {'axes.facecolor': '0.9'})

# ROC curve for training set
plt.figure(figsize=(10, 8))
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
plt.legend(loc='lower right')
print('AUC: {}'.format(auc(train_fpr, train_tpr)))
plt.show()
```


```python
# ROC curve for test set
plt.figure(figsize=(10, 8))
lw = 2

```

What do you notice about these ROC curves? Your answer here: 


```python

```

## Interpret ROC curves

Since the mushroom model is atypically perfect, let's consider another dataset to practice interpreting ROC curves. This heart disease [dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) is widely used in machine learning and a csv file of the data is in the GitHub repository.

Look at the the heart disease's model ROC curve:  

<img src="https://curriculum-content.s3.amazonaws.com/data-science/images/lesson_roc_graph.png">

Think about the scenario of this model: predicting heart disease. If you tune the current model to have an 82% True Positive Rate, (you've still missed 18% of those with heart disease), what is the False positive rate? 


```python
# Write the approximate fpr when tpr = 0.8
fpr = None
```

If you instead tune the model to have a 95.2% True Postive Rate, what will the False Postive Rate be?


```python
# Write the approximate fpr when tpr = 0.95
fpr = None
```

In the case of heart disease dataset, do you find any of the above cases acceptable? How would you tune the model? Describe what this would mean in terms of the number of patients falsely scared of having heart disease and the risk of missing the warning signs for those who do actually have heart disease.

Your answer here: 


```python

```

## Summary

In this lab you further explored ROC curves and AUC, drawing graphs and then interpreting these results to lead to a more detailed and contextualized understanding of your model's accuracy.
