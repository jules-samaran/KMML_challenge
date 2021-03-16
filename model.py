import numpy as np
import cvxpy as cp
from kernel import kernel_function

class KRR:
    def __init__(self, k_name, lamb):
        self.lamb = lamb
        self.X = None
        self.k_name = k_name
        self.K = None
        self.alpha = None

    def fit(self, X, y):
        # Problem data.
        self.X = X
        self.K = kernel_function(self.k_name, self.X, self.X)
        n = self.K.shape[0]

        # Construct the problem.
        alpha = cp.Variable((n,1))
        err = (1/n) * cp.sum_squares(self.K @ alpha - y)
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
    y_test = np.where(logit_test > 0, 1, - 1)

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

def main():
    test_KRR()

if __name__ == "__main__":
    main()






