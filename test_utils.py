from crossval import get_param_list, get_accuracy, cross_validation, grid_search_cv
from model import KRR, SVM


def test_get_param_list():
    param_grid = {'a': [1, 2], 'b': [True, False]}
    test_list = [
        {'a': 1, 'b': True},
        {'a': 1, 'b': False},
        {'a': 2, 'b': True},
        {'a': 2, 'b': False},
    ]
    output_list = get_param_list(param_grid)
    assert np.all(test_list == output_list), 'Problem when getting all parameters lists'


def generate_samples():
    n = 100
    beta = np.random.randn(10)
    X = np.random.randn(n, 10)
    logit = X @ beta
    y = np.where(logit > 0, 1, -1)

    X_test = np.random.randn(10, 10)
    logit_test = X_test @ beta
    y_test = np.where(logit_test > 0, 1, - 1)

    return X, y, X_test, y_test


def test_get_accuracy():
    X, y, X_test, y_test = generate_samples()
    lamb = 1
    svm = SVM(k_name="linear", lamb=lamb)
    svm.fit(X, y)
    acc = get_accuracy(svm, X, y)
    print(acc)
    assert 0 <= acc <= 1, 'accuracy is not between 0 and 1'


def test_cross_validation():
    X, y, X_test, y_test = generate_samples()
    param_dict = {'k_name': 'linear', 'lamb': 1}
    acc = cross_validation(SVM, param_dict, X, y, 5)
    print(acc)
    assert 0 <= acc <= 1, 'accuracy is not between 0 and 1'


def test_grid_search_cv():
    X, y, X_test, y_test = generate_samples()
    param_grid = {'k_name': ['linear'], 'lamb': [1, 10, 100]}
    best_score, best_param = grid_search_cv(SVM, param_grid, X, y, 5)
    print(best_score, best_param)


def main():
    test_get_param_list()
    test_get_accuracy()
    test_cross_validation()
    test_grid_search_cv()


if __name__ == "__main__":
    main()
