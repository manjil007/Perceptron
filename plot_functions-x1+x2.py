import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

x1 = np.linspace(-100, 100, 100)
x2 = x1

X1, X2 = np.meshgrid(np.linspace(-100, 100, 100), np.linspace(-100, 100, 100))
plane_boundary = -X1 + X2

plt.figure(figsize=(10, 8))

plt.plot(x1, x2, 'k-', label='Hyperplane: -x1 + x2 = 0', linewidth=2)

plt.contourf(X1, X2, plane_boundary, levels=[-1000, 0, 1000], colors=['#FFAAAA', '#AAAAFF'], alpha=0.5)

data = pd.read_csv('data/case1.csv', header=None)

x1_sample = data.iloc[:, 0]
x2_sample = data.iloc[:, 1]
labels = data.iloc[:, -1]
labels = labels.astype(int)

print(labels)

x1_positive = x1_sample[labels == 1]
x2_positive = x2_sample[labels == 1]

x1_negative = x1_sample[labels == -1]
x2_negative = x2_sample[labels == -1]

plt.scatter(x1_positive, x2_positive, color='red')
plt.scatter(x1_negative, x2_negative, color='blue')

plt.xlabel('$x1$')
plt.ylabel('$x2$')
plt.title('Hyperplane: $-x_1 + x_2 = 0$ (or $x_1 = x_2$)')
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.legend()

plt.grid(True)
plt.show()
