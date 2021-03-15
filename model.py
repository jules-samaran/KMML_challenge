import numpy as np
import cvxpy as cp
from kernel import kernel_function


class SVM:

    def __init__(self, k_name, lamb):
        self.alpha = None
        self.k_name = k_name
        self.lamb = lamb
        self.X = None

    def fit(self, X, y):
        n = X.shape[0]

        # initialize
        K = kernel_function(self.k_name, X, X)

        # optimize
        alpha = cp.Variable(n)
        obj = cp.Maximize(2 * alpha@y - cp.quad_form(alpha, K))
        constraints = [- cp.multiply(y, alpha) <= 0, cp.multiply(y, alpha) <= 1/(2 * n * self.lamb)]
        prob = cp.Problem(obj, constraints)
        prob.solve()
        self.alpha = alpha.value
        self.X = X

    def predict(self, X_test):
        K = kernel_function(self.k_name, X_test, self.X)
        output = K@self.alpha
        return np.where(output > 0, 1, - 1)


def test_svm():
    n = 100
    lamb = 1 / (2 * n)
    beta = np.random.randn(10)
    X = np.random.randn(n, 10)
    logit = X @ beta
    y = np.where(logit > 0, 1, -1)

    X_test = np.random.randn(10, 10)
    logit_test = X_test @ beta
    y_test = np.where(logit_test > 0, 1, - 1)

    svm = SVM("linear", lamb=lamb)
    svm.fit(X, y)
    y_pred = svm.predict(X_test)

    assert (y_pred == y_test).all()


def main():
    test_svm()


if __name__ == "__main__":
    main()
