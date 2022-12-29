## Chapter 7. Computer Science

### 7.1 Basics
1. [E] Explain supervised, unsupervised, weakly supervised, semi-supervised, and active learning.
```text
- Supervised learning is a type of machine learning where the model is trained on labeled data, meaning that the data used to train the model is already tagged with correct output labels. The model makes predictions based on this labeled data and the goal is for the model to make predictions on new, unseen data that is drawn from the same distribution as the training data. Examples of supervised learning include classification tasks, such as identifying spam emails, and regression tasks, such as predicting the price of a house given its characteristics.

- Unsupervised learning is a type of machine learning where the model is not given any labeled training data and must discover the underlying structure of the data through techniques such as clustering. The goal of unsupervised learning is to find patterns or relationships in the data, rather than to make predictions on new, unseen data. Examples of unsupervised learning include dimensionality reduction and density estimation.

- Weakly supervised learning is a type of machine learning that is intermediate between supervised and unsupervised learning. In weakly supervised learning, the model is given some labeled data, but the labels are not necessarily complete or accurate. This can be the case when it is expensive or time-consuming to label the data, or when the data is noisy or incomplete. Weakly supervised learning algorithms try to make the most of the available labels to learn about the underlying structure of the data.

- Semi-supervised learning is a type of machine learning that is intermediate between supervised and unsupervised learning. In semi-supervised learning, the model is given a small amount of labeled data and a large amount of unlabeled data. The goal is to use the labeled data to learn about the structure of the data, and then use this learned structure to label the unlabeled data. Semi-supervised learning algorithms try to make the most of the available labeled data and use it to infer the structure of the unlabeled data.

- Active learning is a type of machine learning where the model is able to interact with its environment and request labels for specific data points that it is unsure about. The goal of active learning is to improve the efficiency of the model by only labeling the data that is most valuable for improving the model's performance. This is especially useful when labeling data is expensive or time-consuming, as it allows the model to focus on the most important data points.
```

2. Empirical risk minimization.
   1. [E] What’s the risk in empirical risk minimization?
   ```text
   Empirical risk minimization is a method for minimizing the expected loss of a machine learning model on a given dataset. The goal is to find the model parameters that minimize the average loss over the training data. One risk of empirical risk minimization is overfitting, which occurs when the model is too complex and fits the training data too closely, leading to poor generalization to new, unseen data. Overfitting can occur when the model has too many parameters relative to the size of the training data, or when the training data is noisy or contains outliers. To mitigate the risk of overfitting, it is important to use appropriate model complexity and regularization techniques, such as using a simpler model or adding a regularization term to the loss function. It is also important to evaluate the model's performance on a held-out validation dataset to ensure that it generalizes well to new data. 
   Another risk of empirical risk minimization is the assumption that the training data is representative of the underlying distribution of the data. If the training data is not representative, the model's performance on new, unseen data may be poor. To mitigate this risk, it is important to carefully choose the training data and ensure that it is representative of the distribution of data that the model will be applied to.
    ```
   2. [E] Why is it empirical?
   ```text
   Empirical risk minimization is called empirical because it is based on the empirical distribution of the data, which is the distribution of the data that is actually observed in the training set.
    ```
   3. [E] How do we minimize that risk?
   ```text
   Empirical risk minimization involves finding the model parameters that minimize the average loss over the training data. This can be done using optimization algorithms such as gradient descent, which involves iteratively updating the model parameters in the direction that reduces the loss.
    
   To minimize the empirical risk, the loss function is defined based on the task at hand and the desired properties of the model. For example, in a classification task, the loss function could be the cross-entropy loss, which measures the distance between the predicted class probabilities and the true class labels. In a regression task, the loss function could be the mean squared error, which measures the difference between the predicted and true values.
    ```

3. [E] Occam's razor states that when the simple explanation and complex explanation both work equally well, the simple explanation is usually correct. How do we apply this principle in ML?
```text
In machine learning, Occam's Razor is often applied in the context of model selection, where it suggests that, all else being equal, a simpler model with fewer parameters is generally preferred over a more complex model with more parameters. This is because a simpler model is less likely to overfit the data, and is therefore more likely to generalize well to new, unseen data.

For example, suppose you are trying to build a machine learning model to predict the price of a house based on a set of features such as the size of the house, the number of bedrooms, and the location. If you have a choice between two models, one with a single linear regression model and one with a more complex model such as a neural network, Occam's razor suggests that you should generally prefer the simpler linear regression model, unless the additional complexity of the neural network is justified by the data.

In summary, Occam's razor is a principle that suggests that the simplest explanation for a phenomenon is generally the most likely to be correct. In machine learning, it is often applied in the context of model selection, where it suggests that a simpler model with fewer parameters is generally preferred over a more complex model with more parameters.
```
4. [E] What are the conditions that allowed deep learning to gain popularity in the last decade?
5. [M] If we have a wide NN and a deep NN with the same number of parameters, which one is more expressive and why?
```text
A wide neural network refers to a neural network with a large number of units or neurons in each layer, while a deep neural network refers to a neural network with a large number of layers. In general, a wide neural network is more expressive than a deep neural network with the same number of parameters, because it has more units in each layer and is therefore able to model more complex functions. 
If you have a wide neural network and a deep neural network with the same number of parameters, the wide neural network is generally more expressive because it has a higher model capacity. This is because the wide neural network has more units in each layer and is therefore able to model more complex functions. However, it is important to note that this is not always the case, and the relative expressiveness of a wide versus deep neural network will depend on the specific architecture and the data being modeled.
```
6. [H] The Universal Approximation Theorem states that a neural network with 1 hidden layer can approximate any continuous function for inputs within a specific range. Then why can’t a simple neural network reach an arbitrarily small positive error?
7. [E] What are saddle points and local minima? Which are thought to cause more problems for training large NNs?
8. Hyperparameters.
   1. [E] What are the differences between parameters and hyperparameters?
   2. [E] Why is hyperparameter tuning important?
   3. [M] Explain algorithm for tuning hyperparameters.
   
9. Classification vs. regression.
   1. [E] What makes a classification problem different from a regression problem?
   2. [E] Can a classification problem be turned into a regression problem and vice versa?

10. Parametric vs. non-parametric methods.
    1. [E] What’s the difference between parametric methods and non-parametric methods? Give an example of each method.
    2. [H] When should we use one and when should we use the other?
11. [M] Why does ensembling independently trained models generally improve performance?
12. [M] Why does ensembling independently trained models generally improve performance?
13. [E] Why does an ML model’s performance degrade in production?
14. [M] What problems might we run into when deploying large machine learning models?
15. Your model performs really well on the test set but poorly in production.
    1. [M] What are your hypotheses about the causes?
    2. [H] How do you validate whether your hypotheses are correct?
    3. [M] Imagine your hypotheses about the causes are correct. What would you do to address them?

### 7.2 Sampling and creating training data

1. [E] If you have 6 shirts and 4 pairs of pants, how many ways are there to choose 2 shirts and 1 pair of pants?
2. [M] What is the difference between sampling with vs. without replacement? Name an example of when you would use one rather than the other?
3. [M] Explain Markov chain Monte Carlo sampling.
4. [M] If you need to sample from high-dimensional data, which sampling method would you choose?
5. [H] Suppose we have a classification task with many classes. An example is when you have to predict the next word in a sentence -- the next word can be one of many, many possible words. If we have to calculate the probabilities for all classes, it’ll be prohibitively expensive. Instead, we can calculate the probabilities for a small set of candidate classes. This method is called candidate sampling. Name and explain some of the candidate sampling algorithms.
6. Suppose you want to build a model to classify whether a Reddit comment violates the website’s rule. You have 10 million unlabeled comments from 10K users over the last 24 months and you want to label 100K of them.
   1. [M] How would you sample 100K comments to label?
   2. [M] Suppose you get back 100K labeled comments from 20 annotators and you want to look at some labels to estimate the quality of the labels. How many labels would you look at? How would you sample them?
7. [M] Suppose you work for a news site that historically has translated only 1% of all its articles. Your coworker argues that we should translate more articles into Chinese because translations help with the readership. On average, your translated articles have twice as many views as your non-translated articles. What might be wrong with this argument?
8. [M] How to determine whether two sets of samples (e.g. train and test splits) come from the same distribution?

9. [H] How do you know you’ve collected enough samples to train your ML model?
10. [M] How to determine outliers in your data samples? What to do with them?
11. Sample duplication
    1. [M] When should you remove duplicate training samples? When shouldn’t you?
    2. [M] What happens if we accidentally duplicate every data point in your train set or in your test set?
12. Missing data
    1. [H] In your dataset, two out of 20 variables have more than 30% missing values. What would you do?
    2. [M] How might techniques that handle missing data make selection bias worse? How do you handle this bias?
13. [M] Why is randomization important when designing experiments (experimental design)?

14. Class imbalance.
    1. [E] How would class imbalance affect your model?
    2. [E] Why is it hard for ML models to perform well on data with class imbalance?
    3. [M] Imagine you want to build a model to detect skin legions from images. In your training dataset, only 1% of your images shows signs of legions. After training, your model seems to make a lot more false negatives than false positives. What are some of the techniques you'd use to improve your model?
15. Training data leakage.
    1. [M] Imagine you're working with a binary task where the positive class accounts for only 1% of your data. You decide to oversample the rare class then split your data into train and test splits. Your model performs well on the test split but poorly in production. What might have happened?
    2. [M] You want to build a model to classify whether a comment is spam or not spam. You have a dataset of a million comments over the period of 7 days. You decide to randomly split all your data into the train and test splits. Your co-worker points out that this can lead to data leakage. How?

16. [M] How does data sparsity affect your models?
17. Feature leakage
    1. [E] What are some causes of feature leakage?
    2. [E] Why does normalization help prevent feature leakage?
    3. [M] How do you detect feature leakage?

18. [M] Suppose you want to build a model to classify whether a tweet spreads misinformation. You have 100K labeled tweets over the last 24 months. You decide to randomly shuffle on your data and pick 80% to be the train split, 10% to be the valid split, and 10% to be the test split. What might be the problem with this way of partitioning?
19. [M] You’re building a neural network and you want to use both numerical and textual features. How would you process those different features?
20. [H] Your model has been performing fairly well using just a subset of features available in your data. Your boss decided that you should use all the features available instead. What might happen to the training error? What might happen to the test error?


### 7.3 Objective functions, metrics, and evaluation

1. Convergence.
   1. [E] When we say an algorithm converges, what does convergence mean?
   2. [E] How do we know when a model has converged?
2. [E] Draw the loss curves for overfitting and underfitting.
3. Bias-variance trade-off
   1. [E] What’s the bias-variance trade-off?
   2. [M] How’s this tradeoff related to overfitting and underfitting?
   3. [M] How do you know that your model is high variance, low bias? What would you do in this case?
   4. [M] How do you know that your model is low variance, high bias? What would you do in this case?
4. Cross-validation.
   1. [E] Explain different methods for cross-validation.
   2. [M] Why don’t we see more cross-validation in deep learning?
5. Train, valid, test splits.
   1. [E] What’s wrong with training and testing a model on the same data?
   2. [E] Why do we need a validation set on top of a train set and a test set?
   3. [M] Your model’s loss curves on the train, valid, and test sets look like this. What might have been the cause of this? What would you do?
6. [E] Your team is building a system to aid doctors in predicting whether a patient has cancer or not from their X-ray scan. Your colleague announces that the problem is solved now that they’ve built a system that can predict with 99.99% accuracy. How would you respond to that claim?
7. F1 score.
   1. [E] What’s the benefit of F1 over the accuracy?
   2. [M] Can we still use F1 for a problem with more than two classes. How?
8. Given a binary classifier that outputs the following confusion matrix.
   1. [E] Calculate the model’s precision, recall, and F1.
   2. [M] What can we do to improve the model’s performance?
9. Consider a classification where 99% of data belongs to class A and 1% of data belongs to class B.
   1. [M] If your model predicts A 100% of the time, what would the F1 score be? Hint: The F1 score when A is mapped to 0 and B to 1 is different from the F1 score when A is mapped to 1 and B to 0.
   2. [M] If we have a model that predicts A and B at a random (uniformly), what would the expected F1 be?

10. [M] For logistic regression, why is log loss recommended over MSE (mean squared error)?
11. [M] When should we use RMSE (Root Mean Squared Error) over MAE (Mean Absolute Error) and vice versa?
12. [M] Show that the negative log-likelihood and cross-entropy are the same for binary classification tasks.
13. [M] For classification tasks with more than two labels (e.g. MNIST with 10 labels), why is cross-entropy a better loss function than MSE?
14. [E] Consider a language with an alphabet of 27 characters. What would be the maximal entropy of this language?
15. [E] A lot of machine learning models aim to approximate probability distributions. Let’s say P is the distribution of the data and Q is the distribution learned by our model. How do measure how close Q is to P?
16. MPE (Most Probable Explanation) vs. MAP (Maximum A Posteriori)
    1. [E] How do MPE and MAP differ?
    2. [H] Give an example of when they would produce different results.
17. [E] Suppose you want to build a model to predict the price of a stock in the next 8 hours and that the predicted price should never be off more than 10% from the actual price. Which metric would you use?