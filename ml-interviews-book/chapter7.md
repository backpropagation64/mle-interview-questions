### Chapter 7. Computer Science

#### 7.1 Basics
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

3. [E] How Occam's razor applies in ML?
4. [E] What are the conditions that allowed deep learning to gain popularity in the last decade?
5. [M] If we have a wide NN and a deep NN with the same number of parameters, which one is more expressive and why?
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

