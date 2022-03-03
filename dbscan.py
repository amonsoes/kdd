from sklearn.cluster import DBSCAN
import numpy as np
import string
import matplotlib.pyplot as plt

def dbscan(X, eps, min_s, metric):
    letters = string.ascii_uppercase
    clustering = DBSCAN(eps=eps, min_samples=min_s, metric=metric).fit(X)
    core_samples = clustering.core_sample_indices_
    for e,i in enumerate(clustering.labels_):
        if e in clustering.core_sample_indices_:
            print(f'{letters[e]}:{i} -> CORE')
        else:
            print(f'{letters[e]}:{i}')

def make_array (lst):
    lst = [i for i in lst if i != [-1,-1]]
    X = np.array(lst)
    return X


if __name__ == '__main__':  

    EPS = 1
    MPTS = 2
    METRIC = 'manhattan'

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

    def show_graph():
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

    #show_graph()
    print(dbscan(X,EPS, MPTS, METRIC))
