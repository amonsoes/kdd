import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import string

def plot_dendrogram(model, points, **kwargs):
    letters = string.ascii_uppercase
    labelnum = [letters[i] for i in range(points)]
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix,labels = labelnum, **kwargs,)

def make_array (lst):
    lst = [i for i in lst if i != [-1,-1]]
    X = np.array(lst)
    return X

if __name__ == '__main__':

    DISTANCE = 'manhattan'
    LINKAGE = 'single'


    A = [3, 1]
    B = [2, 2]
    C = [3, 2]
    D = [4, 2]
    E = [2, 3]
    F = [4, 3]
    G = [-1,-1]
    H = [-1,-1]
    I = [-1,-1]
    J = [-1,-1]
    K = [-1,-1]
    L = [-1,-1]
    M = [-1,-1]
    N = [-1,-1]
    O = [-1,-1]
    P = [-1,-1]
    Q = [-1,-1]
    R = [-1,-1]
    S = [-1,-1]
    T = [-1,-1]


    X = make_array([A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T])

    NUMPOINTS = len(X)

    letters = string.ascii_uppercase
    labels = range(0, len(X)+1)
    plt.figure(figsize=(10, 7))
    plt.subplots_adjust(bottom=0.1)
    plt.scatter(X[:,0],X[:,1], label='True Position')

    for label, x, y in zip(labels, X[:, 0], X[:, 1]):
        plt.annotate(
            letters[label],
            xy=(x, y), xytext=(-3, 3),
            textcoords='offset points', ha='right', va='bottom')
    plt.show()

    cluster = AgglomerativeClustering(n_clusters=None, 
                                        affinity=DISTANCE, 
                                        linkage=LINKAGE, 
                                        compute_full_tree=True, 
                                        distance_threshold=0)
    print(cluster.fit_predict(X))
    plot_dendrogram(cluster, NUMPOINTS, truncate_mode=None)