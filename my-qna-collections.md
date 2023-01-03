# Answering ML Interview questions
Answering interesting interview questions from different topics (e.g. Bias and Variance, Regulization, NLP, Neural networks, etc). 

*Disclaimer: Most of these questions will be answered be ChatGPT and must not been considered 100% correct, so please review the answers and if you disagree please add an issue or contact the author.*

## Classical Machine Learning

### Bias and Variance
#### Which ML algorithms are considered of high bias and which of low bias ?
    
    There are several algorithms in machine learning that can be biased, either in the training data or in the way they are designed and implemented. Some examples of algorithms that may be prone to bias include:
    
    1. Decision tree algorithms: These algorithms can be biased if the training data is unbalanced or if the tree is not properly pruned.
    
    2. Naive Bayes classifiers: These algorithms can be biased if the assumption of independence between features is not valid for the problem being solved.
    
    3. Linear regression: Linear regression can be biased if there are outliers in the training data or if the features are not independent.
    
    4. Nearest neighbor algorithms: These algorithms can be biased if the distance metric used to determine the nearest neighbors is not appropriate for the problem being solved.
    
    On the other hand, here are some examples of algorithms that are generally considered to have low bias:
    
    1. Deep neural networks: Deep neural networks, which have many layers and can learn complex relationships in the data, are generally considered to have low bias because they can model a wide range of patterns in the data.
    
    2. Support vector machines: Support vector machines are considered to have low bias because they make predictions based on the support vectors, which are the points in the training data that are closest to the decision boundary.
    
    3. Ensemble methods: Ensemble methods, which combine the predictions of multiple models, are generally considered to have low bias because the errors of the individual models are averaged out.
    
    It's important to keep in mind that all machine learning algorithms have the potential to be biased, and it's important to carefully evaluate the data and the algorithm to ensure that the results are fair and accurate.


#### Which ML algorithms are considered of high variance and which of low variance ?

    In machine learning, variance refers to the amount that the model's predictions vary based on different training data. A model with high variance is sensitive to the specific details of the training data, and may perform poorly on new, unseen data. A model with low variance is not as sensitive to the specific details of the training data, and may perform better on new, unseen data.

    Here are some examples of algorithms that are generally considered to have high variance:
    
    1. Decision trees: Decision trees can have high variance because they are sensitive to the specific details of the training data. If the tree is not properly pruned, it can overfit the training data and have poor generalization performance.
    
    2. Neural networks: Neural networks can have high variance because they can have a large number of parameters, which can lead to overfitting if the model is not regularized.
    
    3. K-nearest neighbors: The K-nearest neighbors algorithm can have high variance if the value of K is not set appropriately.
    
    On the other hand, here are some examples of algorithms that are generally considered to have low variance:
    
    1. Linear regression: Linear regression has low variance because it makes a single, global prediction based on the entire dataset.
    
    2. Logistic regression: Logistic regression has low variance because it makes a single, global prediction based on the entire dataset.
    
    3. Naive Bayes classifiers: Naive Bayes classifiers have low variance because they make independent predictions for each feature, which can reduce overfitting.

#### What are root cause of prediction bias?

    There are several potential causes of bias in machine learning predictions:
    
    1. Bias in the training data: If the training data is biased, the model is likely to learn and reproduce those biases in its predictions.
    
    2. Bias in the model: Some models are designed in a way that makes them prone to bias. For example, decision tree algorithms can be biased if the tree is not properly pruned, and linear regression can be biased if there are outliers in the training data or if the features are not independent.
    
    3. Bias in the evaluation: If the evaluation metrics used to measure the performance of the model are not appropriate for the problem being solved, the model's predictions may be biased. For example, using accuracy as an evaluation metric for a highly imbalanced dataset can lead to biased predictions.
    
    4. Bias in the prediction process: If the process of making predictions is biased, the predictions themselves will be biased. For example, if a model is being used to predict the likelihood of parole for prisoners and the parole board has a bias against granting parole to certain groups of people (e.g. based on race or gender), the predictions made by the model will be biased.

#### What's the trade-off between bias and variance?

    In machine learning, there is a trade-off between bias and variance. Bias refers to the difference between the predicted values and the true values, while variance refers to the amount that the model's predictions vary based on different training data. A model with high bias and low variance is likely to underfit the data, while a model with low bias and high variance is likely to overfit the data.
    
    A model with high bias will make consistently wrong predictions, while a model with high variance will make predictions that vary widely based on the training data. To get the best performance, we want to find a model that strikes a balance between bias and variance, and is able to generalize well to new, unseen data.
    
    One way to trade off bias and variance is to use a model with more complexity, which can capture more patterns in the data but may also be more prone to overfitting. Another way is to use regularization techniques, which can help to reduce overfitting by constraining the model's complexity.


### Over/Under-fitting
#### How would you resolve Overfitting or Underfitting?

    Overfitting and underfitting are common problems in machine learning, and there are several ways to address them:
    
    To reduce overfitting:
    1. Use a simpler model: If a model is overfitting, one way to address this is to use a simpler model with fewer parameters.
    2. Use regularization: Regularization is a technique that can help to prevent overfitting by adding a penalty term to the objective function that the model is trying to optimize.
    3. Use cross-validation: Cross-validation is a technique that involves dividing the training data into multiple folds, training the model on one fold, and evaluating it on the remaining folds. This can help to reduce overfitting because the model is trained and evaluated on different subsets of the data.
    
    To reduce underfitting:
    1. Use more data: If a model is underfitting, one way to address this is to use more data to train the model. This can help the model to learn more complex patterns in the data, and can help to improve generalization performance.
    2. Use feature engineering: Feature engineering is the process of designing and creating new features from the raw data that can help the model to learn more effectively.


### Regulization 


## Deep learning