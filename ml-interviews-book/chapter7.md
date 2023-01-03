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
```text
There are several factors that have contributed to the popularity of deep learning in recent years:

- Increased computational power: Deep learning algorithms require a lot of computational power to train. In the last decade, there has been a significant increase in the availability of powerful hardware, such as graphics processing units (GPUs), which are well-suited for deep learning tasks. This has made it possible to train deep learning models on large datasets.

- Availability of large datasets: In order to train effective deep learning models, large amounts of labeled data are needed. In recent years, there has been an explosion of data available on the internet, as well as an increase in the number of companies and organizations collecting and sharing data. This has made it possible to train deep learning models on a wide range of tasks.

- Improvements in algorithms: There have been significant advances in the development of deep learning algorithms in recent years. Researchers have been able to improve the performance of deep learning models by developing new architectures and training techniques.

- Broader adoption of machine learning: Deep learning is a subfield of machine learning, which has also gained popularity in recent years. The success of deep learning has contributed to the broader adoption of machine learning in a variety of industries and applications.

- Real-world successes: The success of deep learning in a variety of real-world tasks, such as image and speech recognition, has contributed to its popularity. Deep learning models have achieved state-of-the-art performance on many tasks, and this has led to a growing interest in using deep learning for a wide range of applications.
```
5. [M] If we have a wide NN and a deep NN with the same number of parameters, which one is more expressive and why?
```text
A wide neural network refers to a neural network with a large number of units or neurons in each layer, while a deep neural network refers to a neural network with a large number of layers. In general, a wide neural network is more expressive than a deep neural network with the same number of parameters, because it has more units in each layer and is therefore able to model more complex functions. 
If you have a wide neural network and a deep neural network with the same number of parameters, the wide neural network is generally more expressive because it has a higher model capacity. This is because the wide neural network has more units in each layer and is therefore able to model more complex functions. However, it is important to note that this is not always the case, and the relative expressiveness of a wide versus deep neural network will depend on the specific architecture and the data being modeled.
```
6. [H] The Universal Approximation Theorem states that a neural network with 1 hidden layer can approximate any continuous function for inputs within a specific range. Then why can’t a simple neural network reach an arbitrarily small positive error?
```text
The Universal Approximation Theorem does not mean that a neural network with a single hidden layer can reach an arbitrarily small positive error on all tasks. There are several reasons why this may not be possible:

1. Limited capacity: A neural network with a single hidden layer has limited capacity, and may not be able to capture the complexity of the target function if it is very complex. As a result, it may not be able to reach an arbitrarily small positive error.

2. Overfitting: If a neural network is trained on a limited amount of data, it may overfit to the training data and perform poorly on unseen data. This can lead to a higher error on the test set.

3. Local optima: During training, the neural network may get stuck in a local minimum of the loss function, which can result in a suboptimal solution. This can lead to a higher error on the test set.

Lack of sufficient data: In some cases, there may not be enough labeled data available to train a neural network to an arbitrarily small positive error.
```
7. [E] What are saddle points and local minima? Which are thought to cause more problems for training large NNs?
```text

```
8. Hyperparameters.
   1. [E] What are the differences between parameters and hyperparameters?
   ```text
   Parameters are the internal variables of a model that are learned during training. These include the weights and biases of a neural network, which are adjusted based on the training data to optimize the model's performance. Parameters are typically adjusted by the optimization algorithm during training, and they are specific to a particular task or dataset.
    
   Hyperparameters, on the other hand, are external variables that are set before training begins. They control the overall behavior of the model and are not learned during training. Examples of hyperparameters include the learning rate, the batch size, and the number of hidden units in a neural network. Hyperparameters are typically set by the practitioner, and they can significantly influence the model's performance.
    ```
   2. [E] Why is hyperparameter tuning important?
   ```text
   Hyperparameter tuning is an important step in the process of training a deep learning model because it can significantly influence the model's performance. Hyperparameters control the overall behavior of a model, and choosing the right values for them can make a big difference in the model's ability to fit the data and generalize to new situations.

   For example, in a neural network, the learning rate is a hyperparameter that controls the step size that the optimization algorithm takes when adjusting the weights and biases. If the learning rate is too high, the optimization algorithm may overshoot the minimum and oscillate, leading to slow or unstable convergence. If the learning rate is too low, the optimization algorithm may take too long to converge to a good solution. Choosing the right learning rate is therefore critical for successful training.
   
   Hyperparameter tuning can be time-consuming and require a lot of trial and error, but it is generally worth the effort because it can significantly improve the performance of a deep learning model. There are several methods for tuning hyperparameters, including manual search, grid search, and random search. It is also possible to use more advanced methods such as Bayesian optimization or evolutionary algorithms.
   
   Overall, hyperparameter tuning is important in deep learning because it allows practitioners to fine-tune the behavior of a model and improve its performance on a particular task or dataset.
   ```
   3. [M] Explain algorithm for tuning hyperparameters.
   ```text
   There are several algorithms that can be used for tuning hyperparameters in deep learning, including manual search, grid search, random search, and more advanced methods such as Bayesian optimization and evolutionary algorithms. Here is a brief overview of each algorithm:

   Manual search: This is the most basic method for tuning hyperparameters. It involves manually adjusting the values of the hyperparameters and evaluating the model's performance on a validation set. This process can be repeated until satisfactory performance is achieved.
   
   Grid search: This is a systematic method for exploring a range of hyperparameter values. The practitioner specifies a set of values for each hyperparameter, and the algorithm trains a model for each combination of hyperparameter values. The performance of each model is evaluated using a validation set, and the combination of hyperparameters that yields the best performance is selected.
   
   Random search: This method involves sampling random combinations of hyperparameter values and training a model for each combination. The performance of each model is evaluated using a validation set, and the combination of hyperparameters that yields the best performance is selected. This method can be more efficient than grid search because it does not require training a model for every combination of hyperparameter values.
   
   Bayesian optimization: This is a more advanced method that uses Bayesian statistics to model the underlying function that relates the hyperparameters to the model's performance. The algorithm uses this model to iteratively select the most promising combinations of hyperparameter values to try next, based on the previous results.
   
   Evolutionary algorithms: These are optimization algorithms that mimic the process of natural evolution to find the best combination of hyperparameter values. They involve generating a population of random hyperparameter values, evaluating the performance of each individual, and using evolutionary operators such as crossover and mutation to generate new individuals for the next generation. The process is repeated until satisfactory performance is achieved.
   
   Overall, the choice of algorithm for tuning hyperparameters will depend on the specific requirements of the task and the available resources. Some algorithms may be more suitable for certain types of tasks or hyperparameter ranges, and some may be more computationally efficient than others. It is often a good idea to try a few different algorithms to see which one works best for a particular task.
   ```
   
9. Classification vs. regression.
   1. [E] What makes a classification problem different from a regression problem?
   ```text
   In a classification problem, the output variable is a categorical variable, which means that it can take on a finite number of values or categories. For example, a model might be trained to classify email messages as spam or not spam, or to classify images of animals as cats, dogs, or birds. In a classification problem, the goal is to predict the class or category that an input belongs to.

   In a regression problem, the output variable is a continuous variable, which means that it can take on any value within a range. For example, a model might be trained to predict the price of a house given its features (e.g., size, location, number of bedrooms), or to predict the stock price of a company given its financial data. In a regression problem, the goal is to predict a continuous value.

   In summary, the main difference between classification and regression is the type of output variable that is being predicted. Classification involves predicting a categorical variable, while regression involves predicting a continuous variable.
   ```
   2. [E] Can a classification problem be turned into a regression problem and vice versa?
   ```text
   Yes, it is often possible to convert a classification problem into a regression problem and vice versa. Here are some examples of how this can be done:

   Classification to regression: One way to convert a classification problem into a regression problem is to use a method called "ordinal encoding." This involves assigning a numerical value to each class, such that the difference between the values reflects the relative order or importance of the classes. For example, if a model is trained to classify email messages as spam or not spam, the classes could be encoded as 0 (not spam) and 1 (spam). The model could then be trained to predict the encoded values as a continuous output.

   Regression to classification: One way to convert a regression problem into a classification problem is to define a set of threshold values that divide the output range into discrete classes. For example, if a model is trained to predict the price of a house, the output range could be divided into classes such as "cheap," "moderate," and "expensive." The model could then be trained to predict the class that a given input belongs to.

   It is important to note that these conversion methods are not always appropriate and may not produce good results in all cases. The suitability of a particular conversion method will depend on the specifics of the task and the characteristics of the data. In general, it is usually best to use the appropriate type of problem for the task at hand, rather than trying to convert it to a different type.
   ```
10. Parametric vs. non-parametric methods.
    1. [E] What’s the difference between parametric methods and non-parametric methods? Give an example of each method.
    ```text
    In machine learning, parametric methods and non-parametric methods are two types of techniques that can be used to model and make predictions from data.

    Parametric methods are based on the assumption that the data is generated from a specific type of probability distribution with a fixed set of parameters. These methods involve estimating the parameters of the distribution from the data and using them to make predictions. Examples of parametric methods include linear regression, logistic regression, and linear discriminant analysis.

    Non-parametric methods, on the other hand, do not make any assumptions about the underlying distribution of the data. These methods can be more flexible than parametric methods because they do not rely on a fixed set of parameters. Examples of non-parametric methods include decision trees, k-nearest neighbors, and support vector machines.

    In general, parametric methods are typically faster and more computationally efficient than non-parametric methods, but they may be less flexible and may not perform as well on complex or irregular data. Non-parametric methods can be more flexible, but they may require more data and computational resources to train.

    It is important to note that the choice between parametric and non-parametric methods will depend on the specific requirements of the task and the characteristics of the data. In some cases, a parametric method may be the best choice, while in other cases a non-parametric method may be more appropriate.
    ```
    2. [H] When should we use one and when should we use the other?
11. [M] Why does ensembling independently trained models generally improve performance?
    ```text
    Ensembling is a machine learning technique that involves combining the predictions of multiple independently trained models to make a final prediction. Ensembling is often used to improve the performance of a model because it can reduce the variance and bias of the final prediction.

    One reason why ensembling can improve performance is that it can reduce variance. When multiple models are trained independently, they are likely to make different errors due to randomness in the training data. By combining the predictions of these models, the overall error is likely to be reduced because the errors will tend to cancel out. This can lead to more stable and consistent predictions.

    Another reason why ensembling can improve performance is that it can reduce bias. Individual models may have a bias towards certain patterns or features in the data, and this can limit their ability to generalize to new data. By combining the predictions of multiple models, the overall bias is likely to be reduced because the models are likely to have different biases. This can lead to better generalization and improved performance on new data.

    Overall, ensembling is a powerful technique that can improve the performance of a model by reducing variance and bias. It is often used in conjunction with other techniques such as model selection and hyperparameter optimization to further improve the performance of a model.
    ```
12. [M] Why does L1 regularization tend to lead to sparsity while L2 regularization pushes weights closer to 0?
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