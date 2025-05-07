# Company A Questions and Answers (beta)

### Machine Learning
1. Describe Supervised, unsupervised, semi supervised learning with examples.
2. How would you use a model trained with only a few examples that has not optimal scores in order to train an unsupervised model for a similar task?

One approach to using a model trained on a small dataset to train an unsupervised model for a similar task would be to use the model as a feature extractor. A feature extractor is a model that is trained to extract useful features from input data, which can then be used as input to an unsupervised model. Here's how you might use a small, poorly performing supervised model as a feature extractor: 

* Train the small, supervised model on the available labeled data.
* Extract the hidden layer activations of the model when it is presented with new, unlabeled data. These activations can be thought of as a representation of the input data in the feature space of the model.
* Use the extracted features as input to an unsupervised model, such as a clustering algorithm or an autoencoder.
* Train the unsupervised model on the extracted features.
* Using the small, supervised model as a feature extractor can allow you to leverage the knowledge learned by the model, even if it is not performing well on the original task. The unsupervised model can then use the extracted features to discover patterns or relationships in the data that might not be apparent from the raw input data.

It is important to keep in mind that this approach will only work if the supervised model is learning useful features that are relevant to the task at hand. If the model is not learning useful features, the extracted features may not be helpful for training the unsupervised model.

3. Is accuracy a good measure to evaluate a binary classification model?

Accuracy is a commonly used metric to evaluate the performance of a binary classification model, but it is not always the best metric to use. Accuracy is defined as the number of correct predictions made by the model divided by the total number of predictions made. It is a simple and straightforward metric to compute, but it can be misleading if the classes in the dataset are imbalanced. For example, if the dataset contains 99% negative examples and 1% positive examples, a model that always predicts negative would have an accuracy of 99%, even though it is not making any useful predictions. In cases where the classes are imbalanced, it is often more informative to use metrics that take into account the class distribution, such as precision, recall, and the F1 score. These metrics give a more detailed view of the model's performance, and can be more useful for evaluating the model's effectiveness. It is important to consider the context in which the model will be used when choosing an evaluation metric. For example, if the model is being used to predict whether a patient has a certain disease, false negatives (predictions of "no disease" when the patient actually has the disease) may be more concerning than false positives, in which case, recall might be a more important metric to focus on.

4. Let a dataset with thousands of features. How do you decide what feature to keep that are more useful when training a model?
5. Explain how SVM works.
6. Explain L1 and L2 regularization.
7. How can you avoid overfitting?
8. What is the purpose of the activation function?
9. Given 2 sets of inputs x and y calculate the output of relu and sigmoid activation function


The rectified linear unit (ReLU) activation function is defined as f(x) = max(0, x), so for input x=0.5, the output of the ReLU function would be 0.5.
The sigmoid activation function is defined as f(x) = 1 / (1 + e^(-x)), so for input y=2, the output of the sigmoid function would be approximately 0.8807970779778823.

10. Where should we use the sigmoid and where the relu activation functions 

The ReLU activation function is typically used in the hidden layers of a neural network, as it helps to introduce non-linearity and alleviate the vanishing gradient problem. It is also computationally efficient, as it only requires a simple threshold operation. The ReLU is commonly used in image and speech recognition, and natural language processing tasks.

On the other hand, the sigmoid activation function is mostly used in the output layer of a binary classification neural network. Sigmoid function output a probability value between 0 and 1. It is used to predict the probability of an instance belonging to a particular class. Because of this, it's commonly used in logistic regression, where we want to predict the probability of success.

It's important to note that the ReLU activation may not be the best choice when the input data is negative, as it will output zero, so in this cases Leaky ReLU, ELU are some of the alternatives.

11. Explain as detailed as possible the pipeline for training a text classification model (text pre-process, feature extraction , model selection , training process)
12. Let's say you need to create recommendation system for online ads for a sites users. The training dataset consists of user preference and labels data. Explain what type of model will you use and how will you train it?

There are several types of models that could be used to create a recommendation system for online ads, and the choice of model will depend on the specific requirements of the system and the characteristics of the training dataset. Here are some potential approaches:

* Collaborative filtering: Collaborative filtering is a method of making recommendations based on the preferences of similar users. This approach would involve training a model that takes as input a user's past ad clicks and outputs a list of recommended ads. The model could be trained using a matrix factorization algorithm, such as singular value decomposition (SVD), or a neural network.

* Content-based filtering: Content-based filtering is a method of making recommendations based on the characteristics of the items being recommended. This approach would involve training a model that takes as input the characteristics of an ad (e.g., its category, title, and description) and outputs a score indicating how relevant the ad is to a given user. The model could be trained using a supervised learning algorithm, such as a support vector machine (SVM) or a decision tree.

* Hybrid approach: It is also possible to combine the above approaches by using a hybrid model that combines collaborative filtering and content-based filtering. This could involve training separate models for collaborative filtering and content-based filtering and combining their outputs to generate recommendations.

To train the model, you would need to split the dataset into a training set and a validation set. The model would then be trained on the training set and evaluated on the validation set to determine its performance. The model's hyperparameters (e.g., the learning rate, the size of the hidden layers) could be tuned using techniques such as grid search or random search to find the best combination of hyperparameters for the task at hand. Once the model has been trained and validated, it can be deployed in the recommendation system.

13. Let's say that you want to train a Naive Bayes model as a baseline for the online ads recommendation system. How would you train a naive Bayes model if your data consist of millions of classes?

Training a Naive Bayes model with millions of classes can be challenging due to the high dimensional nature of the data and the computational complexity of the model. Here are some potential approaches to training a Naive Bayes model with a large number of classes:

* Use a variant of Naive Bayes that is more suited to high-dimensional data: There are several variants of Naive Bayes that are better suited to handling high-dimensional data than the standard Naive Bayes model. These include the Complement Naive Bayes (CNB) and the Multinomial Naive Bayes (MNB) models. CNB is particularly effective at handling imbalanced datasets, while MNB is better suited to sparse data.

* Use feature selection to reduce the dimensionality of the data: One way to reduce the dimensionality of the data is to use feature selection techniques to identify the most relevant features and discard the rest. This can help to reduce the complexity of the model and make it more tractable to train.

* Use a batch training approach: Instead of training the model on the entire dataset at once, you can split the dataset into smaller batches and train the model on each batch separately. This can help to reduce the memory and computational requirements of the model, making it more feasible to train on a large dataset.

* Use distributed training: If the data is too large to fit on a single machine, you can use distributed training to train the model across multiple machines. This can help to speed up the training process and make it more scalable.

It is important to keep in mind that the performance of a Naive Bayes model may not be competitive with more advanced models on a large, high-dimensional dataset. If the goal is to achieve the highest possible performance, it may be necessary to consider using a different type of model.

14. Why we call Naive Bayes "Naive"?

The Naive Bayes model is called "naive" because it makes a strong assumption about the independence of the features in the data. Specifically, the model assumes that all of the features are independent of each other given the class label. This assumption is often unrealistic in practice, as features can be correlated with each other and with the class label.

For example, consider a dataset of emails that are classified as either spam or not spam. The model might assume that the presence of certain words (e.g., "Viagra") is independent of the presence of other words (e.g., "free"), given the class label. However, in reality, the presence of certain words is often correlated with the presence of other words.

Despite the unrealistic assumption of feature independence, the Naive Bayes model can still be very effective in practice, particularly for classification tasks with a small number of features. This is because the model is simple and easy to implement, and it can perform well with relatively little data.

15. What if you wanted to group two site users together based on their preferences and how will this help build a recommendation model?

We could use a clustering algorithm to group similar users together. Clustering is a technique used to group similar data points together. There are several different clustering algorithms that could be used for this task, such as k-means, hierarchical clustering, or density-based clustering. These algorithms work by analyzing the attributes of the users, such as:  
  - browsing history
  - purchase history
  - demographic information
  - other similar attributes

Once users are grouped together based on their preferences, this information can be used to build a recommendation model. Additionally, clustering can be used to identify patterns and trends in user preferences, which can be used to develop targeted marketing strategies or to improve the overall user experience on the website.

### Deep Learning
16. Why do we need multiple hidden layers?

In deep learning, multiple hidden layers are used to allow the model to learn more complex patterns in the data. A single hidden layer is sufficient to represent any function that can be represented using a combination of simple functions, such as linear and nonlinear transformations. However, as the complexity of the function increases, the number of hidden units required to represent it also increases.

Using multiple hidden layers allows the model to learn more complex patterns by composing simple patterns learned by the lower layers. For example, a model with three hidden layers could learn to recognize edges in the first hidden layer, shapes in the second hidden layer, and objects in the third hidden layer. Each layer builds on the representation learned by the previous layer, allowing the model to learn more abstract and complex patterns.

There are trade-offs to using multiple hidden layers, however. Adding more layers can increase the model's capacity and improve its ability to learn complex patterns, but it can also make the model more prone to overfitting, particularly if the model is not regularized appropriately. It is important to strike a balance between the model's capacity and its generalization ability.

17. What to expect to happen when we add a very deep network of hundreds of layers in a neural network?

Adding a very deep network with hundreds of layers to a neural network can lead to a number of different outcomes, depending on the characteristics of the data and the model architecture. Here are a few possible scenarios:

* Improved performance: In some cases, adding more layers to a neural network can improve its performance on the task at hand. This is because a deeper network can learn more complex patterns in the data, allowing it to make more accurate predictions. However, this improvement in performance is not always guaranteed, and the benefits of adding more layers can diminish as the network becomes deeper.

* Overfitting: Adding more layers to a neural network can also increase the risk of overfitting, particularly if the model is not regularized appropriately. Overfitting occurs when the model performs well on the training data but poorly on unseen data, and it can be caused by the model having too much capacity to fit the noise in the training data.

* Increased training time: Training a very deep network can be computationally intensive, and it may take significantly longer to train compared to a shallower network. This can be a particular issue if the model is being trained on a large dataset or on a resource-constrained machine.

* Decreased training stability: Very deep networks can also be more sensitive to the initialization of the weights and the choice of optimization algorithm. This can make it more difficult to train the model and can lead to unstable training dynamics, such as oscillations or divergence.

Overall, adding a very deep network with hundreds of layers to a neural network can lead to improved performance in some cases, but it is important to carefully consider the trade-offs and ensure that the model is properly regularized and optimized.

18. What to expect to happen when we add a very large input layer in a neural network?

Adding a very large input layer to a neural network can lead to a number of different outcomes, depending on the characteristics of the data and the model architecture. Here are a few possible scenarios:

* Improved performance: In some cases, a larger input layer can improve the performance of the model on the task at hand. This is because a larger input layer can allow the model to consider more features of the input data, which may contain important information that can be used to make more accurate predictions. However, this improvement in performance is not always guaranteed, and the benefits of adding more input units can diminish as the input layer becomes larger.

* Increased training time: Training a model with a very large input layer can be computationally intensive, and it may take significantly longer to train compared to a model with a smaller input layer. This can be a particular issue if the model is being trained on a large dataset or on a resource-constrained machine.

* Overfitting: A larger input layer can also increase the risk of overfitting, particularly if the model is not regularized appropriately. Overfitting occurs when the model performs well on the training data but poorly on unseen data, and it can be caused by the model having too much capacity to fit the noise in the training data.

* Decreased training stability: A model with a very large input layer can also be more sensitive to the initialization of the weights and the choice of optimization algorithm. This can make it more difficult to train the model and can lead to unstable training dynamics, such as oscillations or divergence.

Overall, adding a very large input layer to a neural network can lead to improved performance in some cases, but it is important to carefully consider the trade-offs and ensure that the model is properly regularized and optimized.

19. What if we don't use any activation function in a deep neural network?

If no activation function is used in a deep neural network, the network will simply perform a linear transformation on the input data. Without an activation function, the network will not be able to introduce non-linearity and represent complex relationships between the input and output data. The network will not be able to learn and make predictions on non-linearly separable data, such as images, speech, and natural language.
Additionally, without the activation function the backpropagation algorithm would not be able to compute the gradients and update the weights, causing the network to fail to learn.

20. What are the pros and cons of using BiLSTM or RNN layers instead of deep MLP?
21. Explain what an Auto Encoder is

### Data Structures
21. What type of DS does python use for dictionaries?

Hash maps

22. What are the advantages of using a hash map?
23. What is the time complexity for searching for a value in array?

### Algorithms
24. Code a “find 2 values that sum up to a target in an array” program.

