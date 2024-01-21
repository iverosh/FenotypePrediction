import numpy as np


def myMDS(dist, m = 2):    
    n = dist.shape[0]
    c = np.eye(n) - np.ones((n, n)) / n
    b = np.dot(c, np.dot(dist, c)) * (-0.5)
    eigenvalues, eigenvectors = np.linalg.eig(b)
    indexes = np.argsort(eigenvalues)[-m:]
    eigenvalues = np.array([eigenvalues[index] for index in indexes])
    eigenvectors = np.array([eigenvectors[index] for index in indexes])
    return np.dot(eigenvectors.T, np.diag(eigenvalues ** 0.5))


