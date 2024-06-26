## MNIST Dataset
MNIST is a classification dataset where the target variable is categorical (0 to 9). Hence classification algorithms are used to train this dataset.


## Approach Used

Logistic Regression: This approach is chosen for its simplicity and ease of interpretation, making it an ideal starting point for classification tasks. While it may not capture complex patterns like more advanced models, it provides a basic understanding of the data.

Support Vector Machines (SVMs): This algorithm is suitable for the non-linear and complex datasets. Their ability to capture intricate patterns promises high classification accuracy.

Random Forests: It is an ensemble learning method that can handle complexity in the data. It can avoid focusing too much on small details and give more reliable predictions preventing overfitting.

Convolutional Neural Networks (CNNs): These are ideal for image datasets because they're designed to capture complex relationships within images. With specialized layers like convolutional and pooling layers, CNNs excel at capturing spatial patterns, resulting in exceptional accuracy on datasets like MNIST.

## Conclusion:
Based on the experimental trials on the MNIST dataset, Convolutional Neural Networks (CNNs) emerged as the top-performing model, achieving the highest accuracy, while Logistic Regression, Support Vector Machines, and Random Forests also demonstrated competitive performance, providing alternative options depending on specific requirements and constraints
