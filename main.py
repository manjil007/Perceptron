import numpy as np
from Perceptron import Perceptron
import pandas as pd


def generate_case_one_samples(num_samples=100):
    """
    This function generates samples of class 1 and class 2 as per the requirement stated in
    Case 1. This sample dataset will be used to train and test our model

    :param num_samples: Number of rows in the input dataset
    :return: samples of class 1 and class 2
    """
    class1_samples = []
    class2_samples = []
    while len(class1_samples) < num_samples or len(class2_samples) < num_samples:
        x1 = np.random.uniform(-100, 100)
        x2 = np.random.uniform(-100, 100)

        if (-x1 + x2 > 0) and (len(class1_samples) < num_samples):
            class1_samples.append((x1, x2, 1))
        elif (-x1 + x2 < 0) and (len(class2_samples) < num_samples):
            class2_samples.append((x1, x2, -1))
        else:
            continue

    return class1_samples, class2_samples


def generate_case_two_samples(num_samples=100):
    """
    This function generates samples of class 1 and class 2 as per the requirement stated in
    Case 2. This sample dataset will be used to train and test our model

    :param num_samples: Number of rows in the input dataset
    :return: samples of class 1 and class 2
    """

    class1_samples = []
    class2_samples = []
    while len(class1_samples) < num_samples or len(class2_samples) < num_samples:
        x1 = np.random.uniform(-100, 100)
        x2 = np.random.uniform(-100, 100)

        if (x1 - (2 * x2) + 5 > 0) and (len(class1_samples) < num_samples):
            class1_samples.append((x1, x2, 1))
        elif (x1 - (2 * x2) + 5 < 0) and (len(class2_samples) < num_samples):
            class2_samples.append((x1, x2, -1))
        else:
            continue

    return class1_samples, class2_samples


def generate_case_three_samples(num_samples=100):
    """
    This function generates samples of class 1 and class 2 as per the requirement stated in
    Case 3. This sample dataset will be used to train and test our model

    :param num_samples: Number of rows in the input dataset
    :return: samples of class 1 and class 2
    """

    class1_samples = []
    class2_samples = []
    while len(class1_samples) < num_samples or len(class2_samples) < num_samples:
        x1 = np.random.uniform(-100, 100)
        x2 = np.random.uniform(-100, 100)
        x3 = np.random.uniform(-100, 100)
        x4 = np.random.uniform(-100, 100)
        x5 = np.random.uniform(-100, 100)

        if ((0.5 * x1) - x2 + (10 * x3) + x4 + 50 > 0) and (len(class1_samples) < num_samples):
            class1_samples.append((x1, x2, x3, x4, 1))
        elif ((0.5 * x1) - x2 + (10 * x3) + x4 + 50 < 0) and (len(class2_samples) < num_samples):
            class2_samples.append((x1, x2, x3, x4, -1))
        else:
            continue

    return class1_samples, class2_samples


# Test CASE 1 with fit function of Perceptron
class1_samples_case_1, class2_samples_case_1 = generate_case_one_samples()

# Convert list of tuples to numpy array, merge class 1 and class 2 dataset and shuffle them
np_class1_samples_case_1 = np.array(class1_samples_case_1)
np_class2_samples_case_1 = np.array(class2_samples_case_1)
df_case1 = np.vstack((np_class1_samples_case_1, np_class2_samples_case_1))
np.random.shuffle(df_case1)

np.savetxt('data/case1.csv', df_case1, delimiter=',')

# Separate out features and labels of the dataset for training purpose.
X_1 = df_case1[:, :-1]
y_1 = df_case1[:, -1]

# Initialize perceptron class for Case 1.
perceptron_case_1_no_gradient = Perceptron(2, 2, 0.01)
weight_1, bias_1 = perceptron_case_1_no_gradient.fit(X_1, y_1, 10)

# Generate test dataset for evaluating the model
test_case_1_class_1, test_case_1_class_2 = generate_case_one_samples(10)
test_case_1_class_1, test_case_1_class_2 = np.array(test_case_1_class_1), np.array(test_case_1_class_2)
df_test_case_1 = np.vstack((test_case_1_class_1, test_case_1_class_2))
np.random.shuffle(df_test_case_1)
test_X_1 = df_test_case_1[:, :-1]
test_y_1 = df_test_case_1[:, -1]

# Making prediction and calculating accuracy
prediction_case_1 = perceptron_case_1_no_gradient.forward(test_X_1)
case_1_accuracy = np.sum(prediction_case_1 == test_y_1) / len(test_y_1)
print('case 1 accuracy without gradient = ', case_1_accuracy)

df_test_case_1_with_grad = np.copy(df_test_case_1)

prediction_case_1 = prediction_case_1.reshape(-1, 1)

df_test_case_1 = np.append(df_test_case_1, prediction_case_1, 1)

np.savetxt('data/test_case1.csv', df_test_case_1, delimiter=',')

# With fit_gd
perceptron_case_1_with_gradient = Perceptron(2, 2, 0.01)
weight_1_grad, bias_1_grad = perceptron_case_1_with_gradient.fit_gd(X_1, y_1, 10)
prediction_case_1_with_grad = perceptron_case_1_with_gradient.forward(test_X_1)
case_1_accuracy_with_grad = np.sum(prediction_case_1_with_grad == test_y_1) / len(test_y_1)
print('case 1 accuracy with gradient = ', case_1_accuracy_with_grad)

prediction_case_1_with_grad = prediction_case_1_with_grad.reshape(-1, 1)

df_test_case_1_with_grad = np.append(df_test_case_1_with_grad, prediction_case_1_with_grad, 1)

np.savetxt('data/test_case_1_with_grad.csv', df_test_case_1_with_grad, delimiter=',')

# CASE 2
class1_samples_case_2, class2_samples_case_2 = generate_case_two_samples()
np_class1_samples_case_2 = np.array(class1_samples_case_2)
np_class2_samples_case_2 = np.array(class2_samples_case_2)
df_case2 = np.vstack((np_class1_samples_case_2, np_class2_samples_case_2))
np.random.shuffle(df_case2)

np.savetxt('data/case2.csv', df_case2, delimiter=',')
X_2 = df_case2[:, :-1]
y_2 = df_case2[:, -1]

perceptron_case_2 = Perceptron(2, 2, 0.01)
weight, bias = perceptron_case_2.fit(X_2, y_2, 10)

test_case_2_class_1, test_case_2_class_2 = generate_case_two_samples(10)
test_case_2_class_1, test_case_2_class_2 = np.array(test_case_2_class_1), np.array(test_case_2_class_2)
df_test_case_2 = np.vstack((test_case_2_class_1, test_case_2_class_2))
np.random.shuffle(df_test_case_2)
test_X_2 = df_test_case_2[:, :-1]
test_y_2 = df_test_case_2[:, -1]

prediction_case_2 = perceptron_case_2.forward(test_X_2)
case_2_accuracy = np.sum(prediction_case_2 == test_y_2) / len(test_y_2)
print('case 2 accuracy = ', case_2_accuracy)

prediction_case_2 = prediction_case_2.reshape(-1, 1)

df_test_case_2 = np.append(df_test_case_2, prediction_case_2, 1)

np.savetxt('data/test_case_2.csv', df_test_case_2, delimiter=',')

# CASE 3
# Generating the dataset, converting them to numpy array, merging array of both class and shuffling them for training
class1_samples_case_3, class2_samples_case_3 = generate_case_three_samples()
np_class1_samples_case_3 = np.array(class1_samples_case_3)
np_class2_samples_case_3 = np.array(class2_samples_case_3)
df_case3 = np.vstack((np_class1_samples_case_3, np_class2_samples_case_3))
np.random.shuffle(df_case3)
np.savetxt('data/case3.csv', df_case3, delimiter=',')

# Separate features and labels for training purpose
X_3 = df_case3[:, :-1]
y_3 = df_case3[:, -1]
perceptron_case_3 = Perceptron(4, 2, 0.01)
weight, bias = perceptron_case_3.fit(X_3, y_3, 10)

# Generating dataset for testing, converting them to numpy array, merging array of both class and shuffling
# them for testing
test_case_3_class_1, test_case_3_class_2 = generate_case_three_samples(10)
test_case_3_class_1, test_case_3_class_2 = np.array(test_case_3_class_1), np.array(test_case_3_class_2)
df_test_case_3 = np.vstack((test_case_3_class_1, test_case_3_class_2))
np.random.shuffle(df_test_case_3)
test_X_3 = df_test_case_3[:, :-1]
test_y_3 = df_test_case_3[:, -1]

# Copying array for training and testing with fit_gd
df_test_case_3_with_grad = np.copy(df_test_case_3)

# Making predictions
prediction_case_3 = perceptron_case_3.forward(test_X_3)

# Calculating and printing accuracy
case_3_accuracy = np.sum(prediction_case_3 == test_y_3) / len(test_y_3)
print('case 3 accuracy = ', case_3_accuracy)

# Reshaping the array to merge predictions in the data array
prediction_case_3 = prediction_case_3.reshape(-1, 1)

# Merging predictions to original array
df_test_case_3 = np.append(df_test_case_3, prediction_case_3, 1)

# Saving the output to csv file along with predictions.
np.savetxt('data/test_case3.csv', df_test_case_3, delimiter=',')


perceptron_case_3_with_gradient = Perceptron(4, 2, 0.01)
weight_3_grad, bias_3_grad = perceptron_case_3_with_gradient.fit_gd(X_3, y_3, 10)
prediction_case_3_with_grad = perceptron_case_3_with_gradient.forward(test_X_3)
case_3_accuracy_with_grad = np.sum(prediction_case_3_with_grad == test_y_3) / len(test_y_3)
print('case 3 accuracy with gradient = ', case_3_accuracy_with_grad)

# Reshaping the array to merge predictions in the data array
prediction_case_3_with_grad = prediction_case_3_with_grad.reshape(-1, 1)

# Merging predictions to original array
df_test_case_3_with_grad = np.append(df_test_case_3_with_grad, prediction_case_3_with_grad, 1)

# Saving the output to csv file along with predictions.
np.savetxt('data/test_case3_with_grad.csv', df_test_case_3_with_grad, delimiter=',')














