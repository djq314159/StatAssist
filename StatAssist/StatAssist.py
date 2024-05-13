import random
import math
from sklearn.mixture import GaussianMixture
import torch
import torch.nn as nn
import pandas as pd


class MATH:
    class ALGEBRA:
        @staticmethod
        def polynomial(x: float, coefficients: list) -> float:
            """
            Returns an evaluation of a polynomial.
            :param x: float
            :param coefficients: list, by increasing degree.
            :return: float
            """
            P = 0
            degree = len(coefficients)
            for i in range(degree):
                P += coefficients[i]*pow(x, i)

            return P

        @staticmethod
        def exponential_equation_1(x: float, Args: list=[1, 0]) -> float:
            """
            Returns an evaluation of a basic exponential equation with base e.
            :param x: float, exponent of e.
            :param Args: list, [coefficient of e, constant]; Default: [1, 0].
            :return: float
            """
            A = Args[0]
            C = Args[1]
            return A*math.exp(x) + C

        @staticmethod
        def exponential_equation_2(x: float, Args: list=[1, 2, 0]) -> float:
            """
            Returns an evaluation of a basic general exponential equation.
            :param x: float, exponent of base.
            :param Args: list, [coefficient of base, base, constant]; Default: [1, 2, 0].
            :return: float
            """
            A = Args[0]
            B = Args[1]
            C = Args[2]
            return A*pow(B, x) + C

        @staticmethod
        def log_equation(x: float, Args: list=[1, 0, 10]) -> float:
            """
            Returns an evaluation of a basic logarithmic equation.
            :param x: float
            :param Args: list, [coefficient of log_n(x), constant, base]; Default: [1, 0, 10].
            :return: float
            """
            A = Args[0]
            C = Args[1]
            base = Args[2]

            return A*math.log(x, base) + C

        @staticmethod
        def vector_add(vector_1: list, vector_2: list, add: bool = True) -> list:
            """
            Returns the sum or difference between two vectors.
            :param vector_1: list
            :param vector_2: list
            :param add: bool, Default: True
            :return: list
            """
            new_vector = []
            for element_x, element_y in vector_1, vector_2:
                if add:
                    new_vector.append(element_x + element_y)
                if not add:
                    new_vector.append(element_x - element_y)

            return new_vector

        @staticmethod
        def vector_dot(vector_1: list, vector_2: list) -> float:
            """
            Returns the dot product between two vectors.
            :param vector_1: list
            :param vector_2: list
            :return: float
            """
            dot_sum = 0
            for element_x, element_y in vector_1, vector_2:
                dot_sum += element_x*element_y

            return dot_sum

        @staticmethod
        def vector_cross(vector_1: list, vector_2: list) -> list:
            """
            Returns the cross product between two 3-dimensional vectors.
            :param vector_1: list
            :param vector_2: list
            :return: list
            """
            a = vector_1[1]*vector_2[2] - vector_1[2]*vector_2[1]
            b = vector_1[0]*vector_2[2] - vector_1[2]*vector_2[0]
            c = vector_1[0]*vector_2[1] - vector_1[1]*vector_2[0]

            return [a, b, c]

    class CALCULUS:
        @staticmethod
        def integral_1(func, func_args: list, interval: list) -> float:
            """
            Returns the area under a defined continuous function on an interval.
            :param func: callable function with x as first param.
            :param func_args: list, arguments for callable function.
            :param interval: interval on which the geometric area is to be calculated.
            :return: float
            """
            dx = 0.0001
            area = 0
            x = interval[0]
            while x < interval[1]:
                area += func(x, func_args)*dx
                x += dx

            return area

        @staticmethod
        def integral_2(dataset: list) -> float:
            """
            Returns a geometric area for a discrete set of x and y values.
            :param dataset: list containing two lists, x values and y values.
            :return: float
            """
            area = 0
            x = dataset[0]
            y = dataset[1]
            interval = [min(x), max(x)]
            n = len(x)
            delta_x = (max(x) - min(x))/n

            X = interval[0]
            i = 0
            while X < interval[1]:
                area += y[i]*delta_x
                i += 1

            return area

    class STATS(CALCULUS):
        @staticmethod
        def get_mean(*Data: list) -> list:
            """
            Returns the mean for multiple sets of data
            :param Data: list, contains a list of datasets.
            :return: list, Measure of the average of a dataset.
            """
            means = []
            for dataset in Data:
                temp = 0
                n = len(dataset)
                for datapoint in dataset:
                    temp += datapoint / n

                means.append(temp)
            return means

        @staticmethod
        def get_variance(datasets: list, means: list) -> list:
            """
            Returns the variance for multiple datasets.
            :param datasets: list, contains a list of datasets.
            :param means: list, means for each dataset.
            :return: list, Measure of the 'spread' of a dataset.
            """
            variances = []

            if len(datasets) == len(means):
                for i in range(len(means)):
                    n = len(datasets[i])
                    temp = 0

                    for j in range(n):
                        temp += pow(datasets[i][j] - means[i], 2) / n

                    variances.append(temp)

                return variances

            return ["len(datasets) != len(means)"]

        @staticmethod
        def get_covariance(datasets: list, means: list) -> float:
            """
            Obtains the covariance for two datasets, x and y.
            :param datasets: list, [dataset 'x', dataset 'x'].
            :param means: list, ['x' mean, 'y' mean].
            :return: float, covariance for datasets x and y.
            """
            covariance = 0
            mu_x = means[0]
            mu_y = means[1]
            x = datasets[0]
            y = datasets[1]
            n = len(datasets[0])

            for i in range(n):
                covariance += (x[i] - mu_x) * (y[i] - mu_y) / n

            return covariance

        @staticmethod
        def get_correlation_coefficient(covariance_xy: float, variance_x: float, variance_y: float) -> float:
            """
            Returns the correlation coefficient for dataset x and y.
            :param covariance_xy: float.
            :param variance_x: float.
            :param variance_y: float.
            :return: flaot, measure of correlation between datasets x and y.
            """
            return covariance_xy / math.sqrt(variance_x * variance_y)

        @staticmethod
        def calculate_confidence(data: list, mean: float, variance: float, z: float = 1.96) -> list:
            """
            Calculates the 95% confidence interval for the population.
            :param data: list, the number of datapoints.
            :param mean: float, the average of the data
            :param variance: float, measure of how 'spread-apart' the data is.
            :param z: Z-score associated with the desired confidence level; Default: 1.96 (95%)
            :return: list, Upper bound and lower bound for desired confidence level.
            """
            pop_size = len(data)
            A = z * math.sqrt(variance / pop_size)

            return [mean - A, mean + A]

        @staticmethod
        def get_linreg_slope(*Args: float) -> float:
            """
            Calculates the slope for a linear regression.
            :param Args: (3) -> correlation coefficient, variance_x, variance_y; (2) -> covariance_xy, variance_x
            :return: float, linear regression slope.
            """
            m = 0
            if len(Args) == 3:
                m = Args[0] * math.sqrt(Args[2] / Args[1])

            if len(Args) == 2:
                m = Args[0] / Args[1]

            return m

        @staticmethod
        def normal(x: float, mean_var: list) -> float:
            """
            Returns an evaluation of the normal distribution at x.
            :param x: float.
            :param mean_var: list, mean/variance
            :return: float.
            """
            mu = mean_var[0]
            variance = mean_var[1]

            A = 1 / math.sqrt(2 * math.pi)
            xp = -0.5 * pow(x - mu, 2) / variance

            return A * math.exp(xp)

        def std_normal_prob(self, x: float, mean: float, variance: float) -> float:
            """
            Returns an evaluation of the standard normal distribution at z; given x, mu, and sigma.
            :param x: float.
            :param mean: float
            :param variance, float
            :return: float
            """
            mu = mean

            z = (x - mu) / math.sqrt(variance)
            P = self.integral_1(self.normal, [0, 1], [-10000, z])

            return P

        @staticmethod
        def gamma_dist(x: float, Args: list) -> float:
            """
            Gamma distribution function
            :param x: float, value > 0.
            :param Args: list, [shape param (alpha), scale/stretch param (beta)]; beta > 0
            :return: float
            """
            alpha = Args[1]
            beta = Args[2]

            return pow(beta, alpha)*pow(x, alpha - 1)*math.exp(-beta*x)

        def gamma_prob(self, alpha: float, beta: float, interval: list=[.00001, 10000]) -> float:
            """
            Returns probability on an interval
            :param alpha: float, shape parameter
            :param beta: float, scale/stretch parameter
            :param interval: list, Default: [.00001, 10000]
            :return: float
            """
            P = self.integral_1(self.gamma_dist, [alpha, beta], interval)
            return P

        @staticmethod
        def monte_carlo_1(sample_space: list, success: float, num_trials: int = 10000) -> float:
            """
            Returns probability via monte carlo method; Success is determined by exact value.
            :param sample_space: list, contains all historical outcomes.
            :param success: float, only values equal to this value constitute a success.
            :param num_trials: int, number of times the historical outcomes will be sampled; Default: 1000
            :return: float, probability of observing 'success' parameter.
            """
            num_success = 0
            for trial in range(num_trials):
                if random.choice(sample_space) == success:
                    num_success += 1

            return num_success / num_trials

        @staticmethod
        def monte_carlo_2(sample_space: list, success_threshold: float, num_trials: int = 10000) -> float:
            """
            Returns probability via monte carlo method; Success is determined by a threshold value.
            :param sample_space: list, contains all historical outcomes.
            :param success_threshold: float, values greater than or equal to this value constitute a success.
            :param num_trials: int, number of times the historical outcomes will be sampled; Default: 1000
            :return: float, probability of observing values greater than or equal to 'success_threshold' parameter.
            """
            num_success = 0
            for trial in range(num_trials):
                if random.choice(sample_space) >= success_threshold:
                    num_success += 1

            return num_success / num_trials

        @staticmethod
        def monte_carlo_3(sample_space: list, success_interval: list, num_trials: int = 1000) -> float:
            """
            Returns probability via monte carlo method; Success is determined by a success interval.
            :param sample_space: list, contains all historical outcomes.
            :param success_interval: list, values within this interval constitute a success.
            :param num_trials: int, number of times the historical outcomes will be sampled; Default: 1000.
            :return: float, probability of observing values within the 'success_interval' parameter.
            """
            num_success = 0
            for trial in range(num_trials):
                if success_interval[0] < random.choice(sample_space) < success_interval[1]:
                    num_success += 1

            return num_success / num_trials

        @staticmethod
        def gaussian_mixed_model(data: list, *, n_clusters=1):
            gmm = GaussianMixture(n_components=n_clusters, random_state=31415, max_iter=1000).fit_predict(data)

        @staticmethod
        def ml_linreg_1(dataset_xy, *, slope: float) -> float:
            """
            PyTorch machine learning algorithm for linear regression.
            :param dataset_xy: list, containing two 1-dimensional lists of the same length.
            :param slope: float, known slope of the line.
            :return: float, y-intercept
            """
            x = []
            y = []
            for i in range(len(dataset_xy[0])):
                x.append([dataset_xy[0][1]])
                y.append([dataset_xy[1][1]])

            x = torch.Tensor(x)
            y = torch.Tensor(y)

            b = nn.Parameter(torch.randn(1, 1))
            M = torch.Tensor([slope])
            loss_f = torch.nn.MSELoss(size_average=False)
            optimizer = torch.optim.SGD([b], lr=0.001)

            epoch = 0
            while epoch < 500:
                pred_y = M * x + b
                optimizer.zero_grad()
                loss = loss_f(pred_y, y)
                loss.backward()
                optimizer.step()

                epoch += 1

            return b.item()

        @staticmethod
        def ml_linreg_2(dataset_xy, *, y_intercept):
            """
            PyTorch machine learning algorithm for linear regression.
            :param dataset_xy: list, containing two 1-dimensional lists of the same length.
            :param y_intercept: float, known y-intercept of the line.
            :return: float, slope
            """
            x = []
            y = []
            for i in range(len(dataset_xy[0])):
                x.append([dataset_xy[0][1]])
                y.append([dataset_xy[1][1]])

            x = torch.Tensor(x)
            y = torch.Tensor(y)

            m = nn.Parameter(torch.randn(1, 1))
            B = torch.Tensor([y_intercept])
            loss_f = torch.nn.MSELoss(size_average=False)
            optimizer = torch.optim.SGD([m], lr=0.001)

            epoch = 0
            while epoch < 1500:
                pred_y = m * x + B
                optimizer.zero_grad()
                loss = loss_f(pred_y, y)
                loss.backward()
                optimizer.step()

                epoch += 1

            return m.item()


class ALGORITHMS:
    @staticmethod
    def linear_regression(data: list) -> (float, float):
        """
        Hybrid linear regression algorithm employing statistical analysis and machine learning.
        :param data: list, [x_data, y_data]
        :return: (float, float), slope / y-intercept
        """
        averages = MATH.STATS.get_mean(data[0], data[1])
        variances = MATH.STATS.get_variance([data[0], data[1]], averages)

        covariance_xy = MATH.STATS.get_covariance([data[0], data[1]], averages)
        correlation = MATH.STATS.get_correlation_coefficient(covariance_xy, variances[0], variances[1])

        m = MATH.STATS.get_linreg_slope(correlation, variances[0], variances[1])
        b = MATH.STATS.ml_linreg_1([data[0], data[1]], slope=m)

        return m, b

    @staticmethod
    def correlation_coefficient_matrix(data: list) -> list:
        """
        Correlation coefficient matrix
        :param data: list
        :return: list
        """
        n_cols = len(data)
        n = len(data[0])
        row = [0 for i in range(n_cols)]
        matrix = [row for i in range(n_cols)]

        for i in range(n_cols):
            for j in range(n_cols):
                if i != j:
                    cov = 0
                    means = MATH.STATS.get_mean(data[i], data[j])
                    variances = MATH.STATS.get_variance(data, means)

                    for k in range(n):
                        cov += (data[i][k] - means[0])*(data[j][k] - means[1])/n

                    matrix[i][j] += cov/pow(variances[0]*variances[1], 0.5)

                if i == j:
                    matrix[i][j] = 0

        return matrix
