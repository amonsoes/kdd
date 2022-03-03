import math
import numpy as np

EPS = 0.0000000000000001

def entropy(lst):
    ab = sum(lst)
    total = 0
    for val in lst:
        perc = val/ab
        if perc == 0:
            perc += EPS
        total += perc*math.log2(perc)
    return -1*total

def gini(lst):
    ab = sum(lst)
    total = 0
    for val in lst:
        total += (val/ab)**2
    return 1-total

def mc(lst):
    ab = sum(lst)
    lst = [i/ab for i in lst]
    max_it = max(lst)
    return 1 - max_it

def information_gain(T, Ti):
    ab = sum(T)
    en_T = entropy(T)
    total = 0
    for ls in Ti:
        total += (sum(ls)/ab)*entropy(ls)
    return en_T - total


def em_resp(pi_list, mean_list, var_list, point):
    #enter mat in [a,b,c,d]
    point = np.array(point)
    score_denom = 0
    dic = {}
    for e, (pi, mean, var) in enumerate(zip(pi_list, mean_list, var_list)):

        var_mat = np.array(var).reshape(2,2)
        mean_vec = np.array(mean)
        det = np.linalg.det(var_mat)
        diff = point - mean_vec
        inv = np.linalg.inv(var_mat)

        inv_diff = np.dot(diff,(np.matmul(inv, diff)))
        denom = ((2*math.pi)**2)*det
        score = pi*(1/math.sqrt(denom))*math.exp(-0.5*inv_diff)
        score_denom += score
        dic[str(e)] = {'pi' : pi,
                        'mean': mean_vec,
                        'var': var_mat,
                        'det': det,
                        'inv': inv,
                        'diff': diff,
                        'score': score,
                        'prob': None}
    for k in dic:
        dic[k]['prob'] = dic[k]['score']/score_denom
    for k in dic:
        print(f'\n\ncluster: {k}\nmean : {dic[k]["mean"]} \ndet : {dic[k]["det"]}\n inv{dic[k]["inv"]}, \nscore: {dic[k]["score"]},\n prob: {dic[k]["prob"]}\n')
    return dic


def eval_clusters(C,G):
    con_tab = np.zeros(len(C)* len(G)).reshape(len(C), len(G))
    n = sum([len(i) for i in C])
    for e,ls in enumerate(C):
        for se,xs in enumerate(G):
            sm = 0
            for p in ls:
                if p in xs:
                    sm += 1
            con_tab[e][se] = sm
    sum_c = con_tab.sum(1)
    prob_g = con_tab.sum(0) / n
    prob_c = con_tab.sum(1) / n
    log_prob_c = np.log2(prob_c)
    log_prob_g = np.log2(prob_g)
    H_c = -sum(prob_c*log_prob_c)
    H_g = -sum(prob_g*log_prob_g)
    log_tab = np.zeros(len(C)* len(G)).reshape(len(C), len(G))
    for e,c_ls in enumerate(con_tab):
        for se, c in enumerate(c_ls):
            try:
                log_tab[e][se] = -(c / sum_c[e]) * math.log2(c/[sum_c[e]])
            except:
                log_tab[e][se] = 0
    H_c_g = sum(log_tab.sum(1) * (sum_c / n))
    I_c_g = H_c - H_c_g

    print(f"I(C,G) = {I_c_g}")
    print(f"H(C|G) = {H_c_g}")
    print(f"H(C) = {H_c}")
    print(f'H(G) = {H_g}')
    print(f' C -int- G = {con_tab}')


if __name__ == '__main__':

    # 'IG' = information gain, 'EM' = EM iteration, 'EVAL' = clustering eval
    MODE = 'IG'
    # -- IG --

    if MODE == 'IG':
        T = [5,3]
        T_i = [[4,0],[0,1], [1,1], [0,1]]
        print('INFORMATION GAIN: \n',information_gain(T,T_i))

    
    # -- EM --
    if MODE == 'EM':
        pi_list = [0.3,0.2,0.5]
        mean_list = [[2,2],[5,3],[1,4]]
        var_list = [[3,0,0,3],[2,1,1,4],[16,0,0,4]]
        p = (2.5,3.0)

        print('EM ITERATION: \n',em_resp(pi_list,mean_list,var_list,p))


    # -- Cluster Eval --
    if MODE == 'EVAL':
        C = [['A', 'B', 'C'], ['D'], ['E', 'F']]
        G = [['A', 'B', 'C'], ['E', 'F', 'D']]
        print('CLUSTER EVAL: \n',eval_clusters(C,G))