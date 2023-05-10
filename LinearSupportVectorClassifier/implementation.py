import numpy as np
from scipy.optimize import minimize


# NOTE: follow the docstrings. In-line comments can be followed, or replaced.
#       Hence, those are the steps, but if it does not match your approach feel
#       free to remove.

def linear_kernel(X1, X2):
    """    Matrix multiplication.

    Given two matrices, A (m X n) and B (n X p), multiply: AB = C (m X p).

    Recall from hw 1. Is there a more optimal way to implement using numpy?
    :param X1:  Matrix A
    type       np.array()
    :param X2:  Matrix B
    type       np.array()

    :return:    C Matrix.
    type       np.array()
    """
    # print(X1.shape, X2.shape)
        # Check if the dimensions of the matrices are compatible for multiplication
    # if X1.shape[1] != X2.shape[0]:
    #     raise ValueError("Matrices are not compatible for multiplication.")

    # Perform matrix multiplication using np.dot()
    # C = np.dot(X1, X2)

    return np.matmul(X1,X2.T)


def nonlinear_kernel(X1, X2, sigma=0.5):
    """
     Compute the value of a nonlinear kernel function for a pair of input vectors.

     Args:
         X1 (numpy.ndarray): A vector of shape (n_features,) representing the first input vector.
         X2 (numpy.ndarray): A vector of shape (n_features,) representing the second input vector.
         sigma (float): The bandwidth parameter of the Gaussian kernel.

     Returns:
         The value of the nonlinear kernel function for the pair of input vectors.

     """
    # (Bonus) TODO: implement

    # Compute the Euclidean distance between the input vectors
    # Compute the value of the Gaussian kernel function
    # Return the kernel value
    return None


def objective_function(X, y, a, kernel):
    """
    Compute the value of the objective function for a given set of inputs.

    Args:
        X (numpy.ndarray): An array of shape (n_samples, n_features) representing the input data.
        y (numpy.ndarray): An array of shape (n_samples,) representing the labels for the input data.
        a (numpy.ndarray): An array of shape (n_samples,) representing the values of the Lagrange multipliers.
        kernel (callable): A function that takes two inputs X and Y and returns the kernel matrix of shape (n_samples, n_samples).

    Returns:
        The value of the objective function for the given inputs.
    """
    # Reshape a and y to be column vectors
    a = a.reshape(-1, 1)
    y = y.reshape(-1, 1)

    # Compute the value of the objective function
    # The first term is the sum of all Lagrange multipliers
    # The second term involves the kernel matrix, the labels and the Lagrange multipliers
    # print("line 75")
    # K = kernel(X, X.T)
    # term1 = np.sum(a)
    # term2 = np.sum(a * y * y.T * K)
    # obj_func = term1 - 0.5 * term2
    return np.sum(a) - 0.5*np.sum(y@y.T * kernel(X,X) * (a@a.T))


class SVM(object):
    """
         Linear Support Vector Machine (SVM) classifier.

         Parameters
         ----------
         C : float, optional (default=1.0)
             Penalty parameter C of the error term.
         max_iter : int, optional (default=1000)
             Maximum number of iterations for the solver.

         Attributes
         ----------
         w : ndarray of shape (n_features,)
             Coefficient vector.
         b : float
             Intercept term.

         Methods
         -------
         fit(X, y)
             Fit the SVM model according to the given training data.

         predict(X)
             Perform classification on samples in X.

         outputs(X)
             Return the SVM outputs for samples in X.

         score(X, y)
             Return the mean accuracy on the given test data and labels.
         """

    def __init__(self, kernel=nonlinear_kernel, C=1.0, max_iter=1e3):
        """
        Initialize SVM

        Parameters
        ----------
        kernel : callable
          Specifies the kernel type to be used in the algorithm. If none is given,
          ‘rbf’ will be used. If a callable is given it is used to pre-compute
          the kernel matrix from data matrices; that matrix should be an array
          of shape (n_samples, n_samples).
        C : float, default=1.0
          Regularization parameter. The strength of the regularization is inversely
          proportional to C. Must be strictly positive. The penalty is a squared l2
          penalty.
        """
        self.kernel = kernel
        self.C = C
        self.max_iter = max_iter
        self.a = None
        self.w = None
        self.b = None

    def fit(self, X, y):
        """
        Fit the SVM model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or (n_samples, n_samples)
          Training vectors, where n_samples is the number of samples and n_features
          is the number of features. For kernel=”precomputed”, the expected shape
          of X is (n_samples, n_samples).

        y : array-like of shape (n_samples,)
          Target values (class labels in classification, real numbers in regression).

        Returns
        -------
        self : object
          Fitted estimator.
        """
        n_samples, n_features = X.shape

        # Initialize Lagrange multipliers
        alpha_init = np.zeros(n_samples)

        # Define constraints and bounds
        # constraints = [{'type': 'eq', 'fun': lambda alpha: np.dot(y, alpha)}]
        constraints = ({'type': 'ineq', 'fun': lambda alpha: self.C - alpha},
                       {'type': 'eq', 'fun': lambda alpha: np.dot(y, alpha)})

        bounds = [(0, self.C) for _ in range(n_samples)]

        # Perform optimization
        # res = minimize(lambda a: np.sum(a) - 0.5*np.sum(y@y.T) *self.kernel(X,X) * (a@a.T),
        #                alpha_init,
        #                method='SLSQP',
        #                bounds=bounds,
        #                constraints=constraints,
        #                options={'maxiter': self.max_iter})
        res = minimize(lambda a: -objective_function(X,y,a,self.kernel),
                       x0=np.zeros(n_samples),
                       bounds=bounds,
                       constraints=constraints,
                       options={'maxiter':self.max_iter},
                       method='SLSQP')

        self.a = res.x # alpha: lagrange multipliers

        support_vectors = res.x > 1e-5
        print(support_vectors)
        self.w = np.dot(X.T, y * self.a)
        # self.b = np.mean(self.y - np.sum(self.a*y*X*X[support_vector], axis=1))

        self.b = np.mean(y[support_vectors] - np.dot(X[support_vectors], self.w))
        # sself.b = y - np.dot(self.w, X.T)
        # self.b = np.mean(y[support_vectors - 1] - np.dot(X[support_vectors - 1], self.w))
        # self.b = np.mean(y[res.x > 0] - self.predict(support_vectors))
        # self.b = np.mean(y[res.x > 0] - np.array([self.predict(sv) for sv in support_vectors if self.predict(sv) is not None]))
        # predictions = np.array([self.predict(sv) for sv in support_vectors if self.predict(sv) is not None])
        # if predictions.size == 0:
        #     self.b = 0
        # else:
        #     self.b = np.mean(y[res.x > 0] - predictions)
        return self

    def predict(self, X):
        """
        Perform classification on samples in X.

        For a one-class model, +1 or -1 is returned.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or (n_samples_test, n_samples_train)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
          Class labels for samples in X.
        """
        decision_values = np.dot(X, self.w) + self.b

        # Classify the samples based on the sign of the decision values
        y_pred = np.sign(decision_values)

        return y_pred

    def outputs(X):
        """
        Perform classification on samples in X.

        For a one-class model, +1 or -1 is returned.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or (n_samples_test, n_samples_train)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
          Class labels for samples in X.
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # TODO: replace this with your own classification logic
        y_pred = np.ones(X.shape[0])

        return y_pred

    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.

        In multi-label classification, this is the subset accuracy which is a harsh
        metric since you require for each sample that each label set be correctly
        predicted.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
          Test samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
          True labels for X.

        Return
        ------
        score : float
          Mean accuracy of self.predict(X)
        """
        y_pred = self.predict(X)
        if y.ndim > 1 and y.shape[1] > 1:
            # For multi-label classification, compute subset accuracy
            score = np.mean(np.all(y_pred == y, axis=1))
        else:
            # For single-label classification, compute accuracy score
            score = np.mean(y_pred == y)

        return score
