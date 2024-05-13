# StatAssist

StatAssist is a Python library that provides statistical and mathematical functions as well as built-in algorithms such as function integration and linear regression.

## Installation

You can install StatAssist using pip:

pip install StatAssist

## Usage

Example 1:

```python
import StatAssist
import matplotlib.pyplot as plt

# Define data
x_data = [0, 1, 2, 3, 4, 5]
y_data = [0, 2, 4, 6, 8, 10]

# Get statistical data
averages = stats.get_means(x_data, y_data)
variances = stats.get_variance([x_data, y_data], averages)

covariance_xy = stats.get_covariance([x_data, y_data], averages)
correlation = stats.get_correlation_coefficient(covariance_xy, variances[0], variances[1])

# Get linear regression parameters
m = stat.get_linreg_slope(correlation, variances[0], variances[1])
b = stat.ml_linreg_1([self.x_data, self.y_data], slope=m)

y_reg = [m*x + b for x in x_data]

# Display plot
plt.scatter(x_data, y_data, color='b')
plt.plot(x_data, y_reg, color='r')

plt.show()
```
Example 2:

```python

import StatAssist
import matplotlib.pyplot as plt

lin_reg = algorithm.linear_regression

# Define data
x_data = [0, 1, 2, 3, 4, 5]
y_data = [0, 2, 4, 6, 8, 10]

# Linear regression of data
m, b = lin_reg([x_data, y_data])

y_reg = [m*x + b for x in x_data]

# Display plot
plt.scatter(x_data, y_data, color='b')
plt.plot(x_data, y_reg, color='r')

plt.show()
```


## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
