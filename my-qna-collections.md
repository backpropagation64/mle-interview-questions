# Answering ML Interview questions
Answering interesting interview questions from different topics (e.g. Bias and Variance, Regulization, NLP, Neural networks, etc). 

*Disclaimer: Most of these questions will be answered be ChatGPT and must not been considered 100% correct, so please review the answers and if you disagree please add an issue or contact the author.*

## Classical Machine Learning

### Basics

#### Define probability and likelihood.
    Probability is a measure of how likely an event is to occur, expressed as a number between 0 and 1. For example, the probability of flipping a coin and it landing on heads is 0.5, or 50%.
  
    Likelihood is the conditional probability, that is a measure of how likely it is that a given set of observations or data would occur, given a specific hypothesis or model. Like probability, likelihood is expressed as a number between 0 and 1, with higher values indicating a better fit or more plausible hypothesis.

#### Define Entropy and Information Gain
    Entropy: Randomness of information being processed.

    Information gain is a measure of how much more "informative" a given piece of data is compared to the data that came before it. It is often used in decision tree learning, a method used to build predictive models in the form of a tree-like structure.
    For example, consider a dataset containing information about different types of fruit. One feature of the data might be the color of the fruit, while another might be the shape. If the color feature provides the greatest information gain, it would be chosen as the root node, and the data would be split into different branches based on the different colors of fruit in the dataset.

#### Define Cross-Entropy
    Cross-entropy is a measure of the difference between two probability distributions. It is often used in machine learning as a loss function, which is a way to measure how well a model is performing. In the context of classification tasks, the cross-entropy loss function is used to compare the predicted probabilities of a model to the true probabilities of the labels. The goal is to minimize the cross-entropy loss, which means that the predicted probabilities should be as close as possible to the true probabilities.

    The cross-entropy loss is defined as:
    
    L = - ∑ y' * log(y)
    
    where L is the cross-entropy loss, y' is the true probability distribution, and y is the predicted probability distribution.

#### Define AUC - ROC Curve
    AUC stands for "Area Under the Curve," and ROC stands for "Receiver Operating Characteristic." Together, AUC-ROC is a metric that is used to evaluate the performance of a binary classification model.

    The AUC is the area under the ROC curve. It is a measure of the model's ability to distinguish between positive and negative examples, with a higher AUC indicating a better-performing model. An AUC of 1.0 indicates a perfect classifier, while an AUC of 0.5 indicates a model that is no better than random chance.
    
    ROC curves are commonly used to visualize the performance of a binary classification model and to compare the performance of different models. They are particularly useful when the class distribution is imbalanced, as they allow for the visualization of the tradeoff between the TPR and FPR.

    An ROC curve plots the true positive rate (TPR) on the y-axis and the false positive rate (FPR) on the x-axis.

#### Define Precision and Recall
    In machine learning, precision and recall are two metrics that are used to evaluate the performance of a classification model. They are both used to measure the accuracy of a model, but they do so in different ways.
    
    - Precision is a measure of the proportion of positive predictions that were actually correct. It is calculated as:
      Precision = True Positives / (True Positives + False Positives)

    - Recall, on the other hand, is a measure of the proportion of actual positive examples that were correctly classified. It is calculated as:
      Recall = True Positives / (True Positives + False Negatives)
    

#### What would we maximize in a classification task, precision or recall?
    The choice between precision and recall will depend on the specific needs and goals of the task at hand.

    If the goal is to minimize false positives, it may be more important to have a model with high precision. For example, in a medical diagnosis setting, it is generally more important to avoid false positives (i.e., incorrectly diagnosing a healthy patient as sick) than it is to catch every single sick patient. In this case, a model with high precision would be preferred.
    
    On the other hand, if the goal is to minimize false negatives, it may be more important to have a model with high recall. For example, in a fraud detection setting, it is generally more important to catch as many fraudulent transactions as possible, even if it means that there will be some false positives (i.e., incorrectly flagging a legitimate transaction as fraudulent). In this case, a model with high recall would be preferred.
    
    Ultimately, the choice between precision and recall will depend on the specific needs and goals of the task, and it may be necessary to trade off one for the other to achieve the desired balance.
    
#### What is the F1 score?

    The F1 score is a metric that combines precision and recall. It is calculated as the harmonic mean of precision and recall, with the F1 score reaching its best value at 1 (perfect precision and recall) and worst at 0.
    The F1 score is calculated as:
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    The F1 score is often used as a way to balance precision and recall, especially when it is important to avoid both false positives and false negatives. It is a useful metric to use when you want to balance precision and recall, and it is particularly useful when the classes are imbalanced.
    

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


### Regularization 

## Decision Trees

## Deep learning

### What is an MLP (multi-layer perceptron) and what are MLP's different parts?

    A Multi-Layer Perceptron (MLP) is a type of feedforward artificial neural network. It is called a "multi-layer perceptron" because it is composed of multiple layers of interconnected "perceptrons", which are the basic building blocks of a neural network.
    The different parts of an MLP are:
    
    Input layer: This is the first layer of the network and it receives the input data. It does not perform any computations, but it can be thought of as the starting point for the data's journey through the network.
    
    Hidden layers: These are the layers that come between the input and output layers. They perform computations on the data and pass it along to the next layer. The number of hidden layers can vary depending on the problem and the complexity of the model.
    
    Output layer: This is the final layer of the network, which produces the output. It receives the processed data from the hidden layers and produces the final output.
    
    Neurons: These are the basic building blocks of an MLP. Each neuron receives input from the previous layer, performs a computation on the input, and passes the result along to the next layer.
    
    Synapses: These are the connections between the neurons. Each synapse has a weight that determines the strength of the connection between two neurons. The weights are adjusted during training to optimize the model's performance.
    
    Activation Function: this is a mathematical function that is applied to the output of each neuron. The purpose of the activation function is to introduce non-linearity into the output of the neuron, allowing it to learn more complex patterns in the data. Common activation functions are ReLU, sigmoid, and tanh.
    
    In summary, an MLP is a type of feedforward neural network that is composed of multiple layers of interconnected perceptrons. The different parts of an MLP are the input layer, hidden layers, output layer, neurons, synapses, and activation function.
    

### Activation functions related questions
    (place holder)

### Weight learning (backpropagation)
    (place holder)

### Gradient descent and loss function
    (place holder)

### what are the benefits of mini batching stochastic descent 
    (place holder)

### What is an Autoencoder and of what parts does it consists of?
    (place holder)

### What is the bottleneck in autoencoders
    (place holder)


## Case Studies
#### 1. Let’s say we want to build a model to predict the time spent for a restaurant to prepare food from the moment an order comes in until the order is ready. What kind of model would we build and what features would we use?
    You could use a regression model since regression models are used to predict continuous numerical values. 
    Some potential features to consider could be:
    - Type of cuisine
    - Number of dishes ordered
    - Day of the week and time of day
    - Size of the restaurant
    - Staff experience
    - Kitchen equipment
    - Number of customers
    - Order complexity

#### 2. Imagine you run a pizza franchise, and you run into a problem with a lot of no-shows after customers place their order. What features would you include in a model to try to predict a no-show?
        
    Ypu could use a binary classification model, like Logistic Regression, Support Vector Machine, Random Forest, XGBoost, or a Neural Network.
    Some potential features to consider could be:
    - Customer demographics
    - Order information
    - Order history
    - Payment method
    - Weather
    - Special offers
    - Delivery or pickup
 
#### 3. Say you work at a major credit card company and are given a dataset of 600,000 credit card transactions. Use this dataset to build a fraud detection model.
 
    Building a fraud detection model using a dataset of 600,000 credit card transactions would involve several steps:

      1. Data preparation: The first step is to prepare the data for modeling. This includes cleaning the data, handling missing values, and feature engineering. Examples of features that are relevant to the problem are: 
        - transaction amount, 
        - the merchant category
        - location of the transaction.
      
      2. Exploratory Data Analysis (EDA): The next step would be to perform Exploratory Data Analysis (EDA) on the data to understand the distribution of the data and the relationship between the different features and the target variable (fraud or not fraud). This will help you to identify any patterns or anomalies in the data that could be useful for the model.
      
      3. Split the data: The dataset needs to be split into training and testing sets.
  
      4. Model selection: Next, you would need to choose an appropriate model for the problem. Some popular models for fraud detection include:
        - Support Vector Machines
        - Random Forest 
        - Neural Networks
      
      5. Model training: After selecting the model, you would train it on the training set.
      
      6. Model evaluation: The model would then be evaluated on the test set. You would use metrics such as accuracy, precision, recall, and the F1-score to evaluate the model's performance.
      
      7. Model tuning: If the model's performance is not satisfactory, you could try to improve it by tuning the hyperparameters of the model.
      
      8. Model deployment: Once you have a satisfactory model, you would deploy it in a production environment to start detecting fraud in real-world transactions.
      
      It is worth noting that fraud detection is a challenging task, and it is important to also consider ethical and legal issues related to the use of the data and the model's decisions, as well as to keep track of the model's performance over time and retrain it as needed.
       
       