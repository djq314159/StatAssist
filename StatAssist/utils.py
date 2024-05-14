from stats import statistics


class ALGORITHMS:
    @staticmethod
    def linear_regression(data: list) -> (float, float):
        """
        Hybrid linear regression algorithm employing statistical analysis and machine learning.
        :param data: list, [x_data, y_data]
        :return: (float, float), slope / y-intercept
        """
        averages = statistics.get_mean(data[0], data[1])
        variances = statistics.get_variance([data[0], data[1]], averages)

        covariance_xy = statistics.get_covariance([data[0], data[1]], averages)
        correlation = statistics.get_correlation_coefficient(covariance_xy, variances[0], variances[1])

        m = statistics.get_linreg_slope(correlation, variances[0], variances[1])
        b = statistics.ml_linreg_1([data[0], data[1]], slope=m)

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
                    means = statistics.get_mean(data[i], data[j])
                    variances = statistics.get_variance(data, means)

                    for k in range(n):
                        cov += (data[i][k] - means[0])*(data[j][k] - means[1])/n

                    matrix[i][j] += cov/pow(variances[0]*variances[1], 0.5)

                if i == j:
                    matrix[i][j] = 0

        return matrix

    @staticmethod
    def integral(func, func_args: list, interval: list) -> float:
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
            area += func(x, func_args) * dx
            x += dx

        return area


class MATH:
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
            dot_sum += element_x * element_y

        return dot_sum

    @staticmethod
    def vector_cross(vector_1: list, vector_2: list) -> list:
        """
        Returns the cross product between two 3-dimensional vectors.
        :param vector_1: list
        :param vector_2: list
        :return: list
        """
        a = vector_1[1] * vector_2[2] - vector_1[2] * vector_2[1]
        b = vector_1[0] * vector_2[2] - vector_1[2] * vector_2[0]
        c = vector_1[0] * vector_2[1] - vector_1[1] * vector_2[0]

        return [a, b, c]


algorithms, vector = ALGORITHMS(), MATH()
