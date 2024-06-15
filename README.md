# Banking Churn Prediction

This project aims to predict customer churn in a banking dataset using a neural network. The steps include loading and preprocessing the data, building and training a neural network, and evaluating its performance.

## Imports

First, we import the necessary libraries. These libraries include pandas for data manipulation and analysis, scikit-learn for preprocessing and evaluation metrics, and TensorFlow/Keras for building the neural network.

## Load the Data

Next, we load the dataset `Churn_Modelling.csv` into a pandas DataFrame. This dataset contains information about customers and whether they have churned. We then display the first few rows of the dataset to understand its structure.

## Check for Missing Values

We check if there are any missing values in the dataset. Missing values can affect the performance of the machine learning model, so it is crucial to handle them appropriately.

## Encode Categorical Variables

We encode the categorical variables `Geography` and `Gender` using `LabelEncoder`. Encoding converts categorical values into numerical values that can be used in machine learning models. We also print the mapping of original categories to numerical values to understand the encoding.

## Select Features and Target

We select the features for the model by dropping columns that are not relevant for prediction. These columns include `RowNumber`, `CustomerId`, `Surname`, and `Exited`. The target variable `y` is set as the `Exited` column, which indicates whether the customer has churned.

## Scale the Features

We scale the features using `StandardScaler` to ensure they have a mean of 0 and a standard deviation of 1. Scaling is important as it standardizes the range of the features, which helps in faster convergence of the neural network.

## Split the Data

Next, we split the data into training and testing sets. We use 80% of the data for training and 20% for testing. The `random_state` parameter ensures that the data is split in the same way every time, making the results reproducible.

## Build the Neural Network

We create a sequential neural network and add layers to it. The network consists of:
- A dense layer with 16 neurons and ReLU activation.
- A dropout layer with a 50% dropout rate to prevent overfitting.
- Another dense layer with 8 neurons and ReLU activation.
- A dense output layer with 1 neuron and sigmoid activation for binary classification.

## Compile the Model

We compile the model using the Adam optimizer. The loss function used is binary cross-entropy, which is suitable for binary classification problems. We also track the accuracy metric to evaluate the model's performance.

## Train the Model

We train the model on the training data for 100 epochs with a batch size of 32. During training, we also validate the model using the testing data to monitor its performance and detect overfitting.

## Make Predictions

We predict the target for the testing data and convert the probabilities to binary class labels. A threshold of 0.5 is used to decide the class labels.

## Evaluate the Model

Finally, we evaluate the model's performance using accuracy, classification report, and confusion matrix. Accuracy measures the overall correctness of the model. The classification report provides detailed metrics like precision, recall, and F1-score. The confusion matrix visualizes the performance of the classification by showing the counts of true positives, true negatives, false positives, and false negatives.
