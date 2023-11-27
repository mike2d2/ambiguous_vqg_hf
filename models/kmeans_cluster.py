import numpy as np
from sklearn.cluster import KMeans


class KMeansCluster:
    def __init__(self, penalty_factor=3.0):
        self.kmeans_wrapper_dict = {k: KMeans(n_clusters=k, random_state=12) for k in range(0, 10)}
        self.penalty_factor = penalty_factor
        self.labels_ = None

    def fit(self, vectors):
        scores_data = []
        assignments = []
        all_labels = []
        for k in range(2, vectors.shape[0]):
            kmeans_wrapper = self.kmeans_wrapper_dict[k]    
            # run kmeans
            kmeans = kmeans_wrapper.fit(vectors) 
            all_labels.append(kmeans.labels_)
            centers = kmeans.predict(vectors) 
            num_centers = len(set(centers))
            # intertia is sum of squared distances of samples to their closest cluster center 
            inertia = kmeans.inertia_
            # penalty depends on how many centers you used compared to how many examples you have 
            penalty = self.penalty_factor * (num_centers-1)/ (vectors.shape[0]-1) 
            # we want balanced clusters
            avg = int(vectors.shape[0] / num_centers)
            num_per_center = {c_name: sum([1 for c in centers if c == c_name]) for c_name in centers}
            diffs = [abs(avg - num_per_center[c]) for c in centers]
            difference_penalty = sum(diffs)
            score = inertia + penalty + difference_penalty 

            # score = kmeans.inertia_ + num_centers ** penalty_factor
            # high intertia means dispersed, low means fits well to the number of clusters
            scores_data.append((score, inertia, penalty, difference_penalty, num_centers))

            assignments.append(centers)
            # if you hit zero inertia, no sense in going further 
            if kmeans.inertia_ < 1e-16:
                break

        score_array = np.array(scores_data)
        min_row = np.argmin(score_array[:,0])
        self.labels_ = all_labels[min_row]