import numpy as np
from Perceptron import Perceptron
import pandas as pd


def generate_case_one_samples(num_samples=100):
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


# CASE 1
class1_samples_case_1, class2_samples_case_1 = generate_case_one_samples()
np_class1_samples_case_1 = np.array(class1_samples_case_1)
np_class2_samples_case_1 = np.array(class2_samples_case_1)
df_case1 = np.vstack((np_class1_samples_case_1, np_class2_samples_case_1))
np.random.shuffle(df_case1)
X_1 = df_case1[:, :-1]
y_1 = df_case1[:, -1]

perceptron_case_1 = Perceptron(2, 2, 0.0001)
weight_1, bias_1 = perceptron_case_1.fit(X_1, y_1, 10)

test_case_1_class_1, test_case_1_class_2 = generate_case_one_samples(10)
test_case_1_class_1, test_case_1_class_2 = np.array(test_case_1_class_1), np.array(test_case_1_class_2)
df_test_case_1 = np.vstack((test_case_1_class_1, test_case_1_class_2))
np.random.shuffle(df_test_case_1)
test_X_1 = df_test_case_1[:, :-1]
test_y_1 = df_test_case_1[:, -1]

prediction_case_1 = perceptron_case_1.forward(test_X_1)
case_1_accuracy = np.sum(prediction_case_1 == test_y_1) / len(test_y_1)
print('case 1 accuracy = ', case_1_accuracy)


# CASE 2
class1_samples_case_2, class2_samples_case_2 = generate_case_two_samples()
np_class1_samples_case_2 = np.array(class1_samples_case_2)
np_class2_samples_case_2 = np.array(class2_samples_case_2)
df_case2 = np.vstack((np_class1_samples_case_2, np_class2_samples_case_2))
np.random.shuffle(df_case2)
X_2 = df_case2[:, :-1]
y_2 = df_case2[:, -1]

perceptron_case_2 = Perceptron(2, 2, 0.0001)
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

# CASE 3
class1_samples_case_3, class2_samples_case_3 = generate_case_three_samples()
np_class1_samples_case_3 = np.array(class1_samples_case_3)
np_class2_samples_case_3 = np.array(class2_samples_case_3)
df_case3 = np.vstack((np_class1_samples_case_3, np_class2_samples_case_3))
np.random.shuffle(df_case3)

X_3 = df_case3[:, :-1]
y_3 = df_case3[:, -1]
perceptron_case_3 = Perceptron(4, 2, 0.0001)
weight, bias = perceptron_case_3.fit(X_3, y_3, 10)

test_case_3_class_1, test_case_3_class_2 = generate_case_three_samples(10)
test_case_3_class_1, test_case_3_class_2 = np.array(test_case_3_class_1), np.array(test_case_3_class_2)
df_test_case_3 = np.vstack((test_case_3_class_1, test_case_3_class_2))
np.random.shuffle(df_test_case_3)
test_X_3 = df_test_case_3[:, :-1]
test_y_3 = df_test_case_3[:, -1]

prediction_case_3 = perceptron_case_3.forward(test_X_3)
case_3_accuracy = np.sum(prediction_case_3 == test_y_3) / len(test_y_3)
print('case 3 accuracy = ', case_3_accuracy)












