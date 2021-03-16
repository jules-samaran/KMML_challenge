import numpy as np
import cvxpy as cp
from kernel import kernel_function


class KRR:
    def __init__(self, k_name, lamb=1.):
        self.name = "KRR"
        self.lamb = lamb
        self.X = None
        self.k_name = k_name
        self.alpha = None

    def fit(self, X, y):
        # Problem data.
        self.X = X
        K = kernel_function(self.k_name, self.X, self.X)
        n = K.shape[0]

        # Construct the problem.
        alpha = cp.Variable((n,1))
        err = (1/n) * cp.sum_squares(K @ alpha - y)
        reg = self.lamb * cp.quad_form(alpha, self.K)
        obj = err + reg
        objective = cp.Minimize(obj)
        problem = cp.Problem(objective)

        # Solve the problem.
        problem.solve()
        self.alpha = alpha.value

    def predict(self, X_test):
        K = kernel_function(self.k_name, X_test, self.X)
        pred = K @ self.alpha
        pred = np.where(pred > 0, 1, -1)
        return pred


def test_KRR():
    # Generate simulated data
    n = 10
    tol = 1e-3

    X = np.random.randn(n, 10)
    beta = np.random.randn(10)
    logit = X @ beta
    y = np.where(logit > 0, 1, -1)
    y = y.reshape((y.shape[0], 1))

    X_test = np.random.randn(10, 10)
    logit_test = X_test @ beta

    # Run KRR and compare with analytical solution
    for lamb in [1, 10, 100]:
        krr = KRR("linear", lamb=lamb)
        krr.fit(X, y)
        try:
            _ = krr.predict(X_test)
        except:
            raise Error('Problem with predict function.')
        alpha_opt = krr.alpha
        alpha_ana = np.linalg.inv(krr.K + lamb * n * np.eye(n)) @ y
        err = np.linalg.norm(alpha_ana - alpha_opt)

        assert err < tol, f'solution to far away from the real one {err:.3f}'

        print(f'Test for lambda={lamb} ok')


class SVM:

    def __init__(self, k_name, lamb=1.):
        self.name = "SVM"
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
    test_KRR()
    test_svm()


if __name__ == "__main__":
    main()
