### Chapter 7. Computer Science

#### 7.1 Basics
1. [E] Explain supervised, unsupervised, weakly supervised, semi-supervised, and active learning.
```text
Supervised learning is a type of machine learning where the model is trained on labeled data, meaning that the data used to train the model is already tagged with correct output labels. The model makes predictions based on this labeled data and the goal is for the model to make predictions on new, unseen data that is drawn from the same distribution as the training data. Examples of supervised learning include classification tasks, such as identifying spam emails, and regression tasks, such as predicting the price of a house given its characteristics.

Unsupervised learning is a type of machine learning where the model is not given any labeled training data and must discover the underlying structure of the data through techniques such as clustering. The goal of unsupervised learning is to find patterns or relationships in the data, rather than to make predictions on new, unseen data. Examples of unsupervised learning include dimensionality reduction and density estimation.

Weakly supervised learning is a type of machine learning that is intermediate between supervised and unsupervised learning. In weakly supervised learning, the model is given some labeled data, but the labels are not necessarily complete or accurate. This can be the case when it is expensive or time-consuming to label the data, or when the data is noisy or incomplete. Weakly supervised learning algorithms try to make the most of the available labels to learn about the underlying structure of the data.

Semi-supervised learning is a type of machine learning that is intermediate between supervised and unsupervised learning. In semi-supervised learning, the model is given a small amount of labeled data and a large amount of unlabeled data. The goal is to use the labeled data to learn about the structure of the data, and then use this learned structure to label the unlabeled data. Semi-supervised learning algorithms try to make the most of the available labeled data and use it to infer the structure of the unlabeled data.

Active learning is a type of machine learning where the model is able to interact with its environment and request labels for specific data points that it is unsure about. The goal of active learning is to improve the efficiency of the model by only labeling the data that is most valuable for improving the model's performance. This is especially useful when labeling data is expensive or time-consuming, as it allows the model to focus on the most important data points.
```

