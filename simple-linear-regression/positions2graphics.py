# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("positions2.csv")
print(data.columns)
level = data["Level"].values.reshape(-1,1)
salary = data["Salary"].values.reshape(-1,1)
regression  = LinearRegression()
regression.fit(level,salary)
print(regression.predict([[8.3]]))
plt.scatter(level,salary,color="red")
plt.plot(level,regression.predict(level),color="blue")
plt.show()


