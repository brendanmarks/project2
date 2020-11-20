"""
Optimizer class implementing minibatch gradient descent for softmax regression. Can utilize either
    gradient descent with momentum, or Adaptive Momentum Estimation (Adam), with optional L1 or L2
    regularization.
"""
# Imports
import numpy as np
from sklearn.utils import shuffle


# Helper functions

# Softmax of input z
def softmax(z):
    return np.exp(z) / sum(np.exp(z))


# Randomizes the order of the (x,y) instances, and outputs batch_size of them
def minibatch(x, y, batch_size):
    x, y = shuffle(x, y)
    if not batch_size:
        return x, y
    else:
        return x[0:batch_size], y[0:batch_size]


class GradientDescent:

    """
    Class fields:
       alphaa - learning rate of the optimizer
       beta1 - momentum hyperparameter
       max_iterations - gradient descent termination condition: maximum times iterated
       epsilon - gradient descent termination condition: minimum gradient size
       batch_size - size of the minibatch to use (default value of 0 indicates use full batch)
       adaptive - if true optimizer uses Adam (Adaptive Moment Estimation) rather than gradient
                    descent with momentum
       beta2 - 2nd hyperparameter for Adam (if using)
       regularize - determines regularization used (if any): 0 indicates no regularization, 1 or 2
                      indicate L1 or L2 regularization respectively
       lambdaa - regularization coefficient if used
       keep_weights - indicates whether to store weights at each iteration
    """
    # Constructor
    def __init__(self, alphaa=0.01, beta1=0.9, max_iterations=1e4, epsilon=1e-8, batch_size=0,
                 adaptive=False, beta2=0.999, regularize=0, lambdaa=0.1, keep_weights=False):

        self.alphaa = alphaa
        self.beta1 = beta1
        self.max_iterations = max_iterations
        self.epsilon = epsilon
        self.batch_size = batch_size

        self.adaptive = adaptive
        self.beta2 = beta2

        self.regularize = regularize
        self.lambdaa = lambdaa

        self.keep_weights = keep_weights
        if keep_weights:
            self.weight_history = []

    # Run method - delegates work to one of 2 helper methods, dependent on if Adam is being used or not
    def run(self, x, y, w):
        if self.adaptive:
            return self.adam(x, y, w)
        else:
            return self.momentum(x, y, w)

    # Gradient Descent with Momentum
    def momentum(self, x, y, w):
        grad = np.inf
        t = 1
        delta_w = 0

        while np.linalg.norm(grad) > self.epsilon and t < self.max_iterations:
            x_mini, y_mini = minibatch(x, y, self.batch_size)
            grad = self.gradient(x_mini, y_mini, w)

            delta_w = (self.beta1 * delta_w) + ((1 - self.beta1) * grad)
            w -= self.alphaa * delta_w

            if self.keep_weights:
                self.weight_history.append(w)
            t += 1
        return w

    # Adaptive Moment Estimation
    def adam(self, x, y, w):
        grad = np.inf
        t = 1
        m = 0
        s = 0

        while np.linalg.norm(grad) > self.epsilon and t < self.max_iterations:
            x_mini, y_mini = minibatch(x, y, self.batch_size)
            grad = self.gradient(x_mini, y_mini, w)

            m = (self.beta1 * m) + ((1 - self.beta1) * grad)
            s = (self.beta2 * s) + ((1 - self.beta2) * np.power(grad, 2))
            mh = m / (1 - np.power(self.beta1, t))
            sh = s / (1 - np.power(self.beta2, t))
            w -= self.alphaa * mh * grad / (np.sqrt(sh) + self.epsilon)

            if self.keep_weights:
                self.weight_history.append(w)
            t += 1
        return w

    # Helper method to calculate gradient (and add regularization penalty if any)
    def gradient(self, x, y, w):
        n, d = x.shape
        yh = softmax(np.dot(x, w))
        grad = np.dot(x.T, yh - y) / n
        if self.regularize == 1:
            grad[1:] += self.lambdaa * np.sign(w[1:])
        elif self.regularize == 2:
            grad[1:] += self.lambdaa * w[1:]
        return grad
