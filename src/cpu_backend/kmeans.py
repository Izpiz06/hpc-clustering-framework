import numpy as np

class KMeansCPU:
    """
    Baseline CPU implementation of the K-Means clustering algorithm using NumPy.
    This serves as the benchmark target for GPU-accelerated backends.
    """

    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=None):
        """
        Initialize the K-Means model.

        Parameters:
        -----------
        n_clusters : int
            The number of clusters to form as well as the number of centroids to generate.
        max_iter : int
            Maximum number of iterations of the k-means algorithm for a single run.
        tol : float
            Relative tolerance with regards to Frobenius norm of the difference 
            in the cluster centers of two consecutive iterations to declare convergence.
        random_state : int, optional
            Determines random number generation for centroid initialization.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.n_iters_ = 0

    def fit(self, X):
        """
        Compute k-means clustering.

        Parameters:
        -----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Training instances to cluster.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # 1. Initialize centroids randomly from the data points
        random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = X[random_indices]

        for i in range(self.max_iter):
            self.n_iters_ += 1
            
            # 2. Assign labels based on closest centroid (Euclidean distance)
            # We use broadcasting to calculate distances efficiently
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            self.labels = np.argmin(distances, axis=1)

            # 3. Update centroids
            new_centroids = np.zeros((self.n_clusters, X.shape[1]))
            for k in range(self.n_clusters):
                # Get all points assigned to cluster k
                cluster_points = X[self.labels == k]
                # If a cluster is empty, re-initialize its centroid randomly
                if len(cluster_points) == 0:
                    new_centroids[k] = X[np.random.choice(X.shape[0])]
                else:
                    new_centroids[k] = cluster_points.mean(axis=0)

            # 4. Check for convergence
            centroid_shift = np.linalg.norm(self.centroids - new_centroids)
            if centroid_shift < self.tol:
                break
                
            self.centroids = new_centroids

        return self

    def predict(self, X):
        """
        Predict the closest cluster each sample in X belongs to.
        """
        if self.centroids is None:
            raise ValueError("The model has not been fitted yet. Call 'fit' first.")
            
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def fit_predict(self, X):
        """
        Compute cluster centers and predict cluster index for each sample.
        """
        return self.fit(X).labels