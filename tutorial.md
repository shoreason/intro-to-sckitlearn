# Linear Regression

```python
# Code source: Jaques Grobler
# License: BSD 3 clause
```
## Import dependencies

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
```
--

```python
## Load the diabetes dataset
diabetes = datasets.load_diabetes()
```
---

## Use only one feature

```python
diabetes_X = diabetes.data[:, np.newaxis, 2]
```
--

```python
## Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]
```

--

## Split the targets into training/testing sets

```python
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]
```

--

## Create linear regression object

```python
regr = linear_model.LinearRegression()
```

--

## Train the model using the training sets

```python
regr.fit(diabetes_X_train, diabetes_y_train)
```

--

## Make predictions using the testing set

```python
diabetes_y_pred = regr.predict(diabetes_X_test)
```

--

## The coefficients

```python
print('Coefficients: \n', regr.coef_)
```

--

## The mean squared error

```python
print("Mean squared error: %.2f"
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
```

--

## Explained variance score: 1 is perfect prediction

```python
print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))
```

--

## Plot outputs

```python
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
```
