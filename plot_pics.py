import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

"""
#mean, cov = [20, 0.5], [(10, 40), (0, 1)]
#data = np.random.multivariate_normal(mean, cov, 200)
#x_sampl = np.random.uniform(10, 40, size=2000)
x_sampl = np.random.normal(loc=0.5, scale=0.3, size=2000)
y_sampl = np.random.normal(loc=0.5, scale=0.3, size=2000)
x_sampl = np.asarray(list(map(lambda x: 0.5 if x < 0 or x > 1 else x, x_sampl)))
y_sampl = np.asarray(list(map(lambda y: 0.5 if y < 0 or y > 1 else y, y_sampl)))
#y_sampl[-1] = 3
#y_sampl[-2] = -3
data = np.zeros((2000, 2))
data[:, 0] = x_sampl
data[:, 1] = y_sampl
#print(data)
df = pd.DataFrame(data, columns=["x", "y"])
sns.jointplot(x="x", y="y", data=df, kind="kde")
#plt.plot(x_sampl, y_sampl)
plt.show()
"""