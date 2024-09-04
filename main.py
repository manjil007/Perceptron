import numpy as np
from Perceptron import Perceptron
import pandas as pd


def generate_case_one_samples(num_samples=100):
    class1_samples = []
    class2_samples = []
    while len(class1_samples) < 100 or len(class2_samples) < 100:
        x1 = np.random.uniform(-100, 100)
        x2 = np.random.uniform(-100, 100)

        if (-x1 + x2 > 0) and (len(class1_samples) < 100):
            class1_samples.append((x1, x2, 1))
        elif (-x1 + x2 < 0) and (len(class2_samples) < 100):
            class2_samples.append((x1, x2, 2))
        else:
            continue

        print('len class1: ', len(class1_samples))
        print('len class2: ', len(class2_samples))

    return class1_samples, class2_samples


def generate_case_two_samples(num_samples=100):
    class1_samples = []
    class2_samples = []
    while len(class1_samples) < 100 or len(class2_samples) < 100:
        x1 = np.random.uniform(-100, 100)
        x2 = np.random.uniform(-100, 100)

        if (x1 - (2 * x2) + 5 > 0) and (len(class1_samples) < 100):
            class1_samples.append((x1, x2, 1))
        elif (x1 - (2 * x2) + 5 < 0) and (len(class2_samples) < 100):
            class2_samples.append((x1, x2, 2))
        else:
            continue

        print('len class1: ', len(class1_samples))
        print('len class2: ', len(class2_samples))

    return class1_samples, class2_samples


def generate_case_three_samples(num_samples=100):
    class1_samples = []
    class2_samples = []
    while len(class1_samples) < 100 or len(class2_samples) < 100:
        x1 = np.random.uniform(-100, 100)
        x2 = np.random.uniform(-100, 100)
        x3 = np.random.uniform(-100, 100)
        x4 = np.random.uniform(-100, 100)
        x5 = np.random.uniform(-100, 100)

        if ((0.5 * x1) - x2 + (10 * x3) + x4 + 50 > 0) and (len(class1_samples) < 100):
            class1_samples.append((x1, x2, x3, x4, 1))
        elif ((0.5 * x1) - x2 + (10 * x3) + x4 + 50 < 0) and (len(class2_samples) < 100):
            class2_samples.append((x1, x2, x3, x4, 2))
        else:
            continue

        print('len class1: ', len(class1_samples))
        print('len class2: ', len(class2_samples))

    return class1_samples, class2_samples


class1_samples, class2_samples = generate_case_one_samples()
np_class1_samples = np.array(class1_samples)
np_class2_samples = np.array(class2_samples)
df_case1 = np.vstack((np_class1_samples, np_class2_samples))
np.random.shuffle(df_case1)

class1_samples_2, class2_samples_2 = generate_case_two_samples()
np_class1_samples_2 = np.array(class1_samples_2)
np_class2_samples_2 = np.array(class2_samples_2)
df_case2 = np.vstack((np_class1_samples_2, np_class2_samples_2))
np.random.shuffle(df_case2)

class1_samples_3, class2_samples_3 = generate_case_three_samples()
np_class1_samples_3 = np.array(class1_samples_3)
np_class2_samples_3 = np.array(class2_samples_3)
df_case3 = np.vstack((np_class1_samples_3, np_class2_samples_3))
np.random.shuffle(df_case3)

perceptron = Perceptron(2, 2, 0.01)

X_1 = df_case1[:, :-1]
y_1 = df_case1[:, -1]

weight, bias = perceptron.fit(X_1, y_1, 10)


test_case = np.array([4, -9])
prediction = perceptron.forward(test_case)
print(prediction)







