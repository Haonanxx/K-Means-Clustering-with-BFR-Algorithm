import time
import sys
import glob
import random
import math
from collections import OrderedDict
from itertools import combinations
import csv
import json

# needs a class KMeans to serve as the in-memory clustering algorithm
# should be functionally similar to sklearn.cluster.KMeans
# also reference to sklearn.cluster.KMeans
class KMeans:
    # initialize
    '''
    parameters:
    K: number of clusters
    seed: random seed (want to be reproduceable for testing convenience)
    tol: for determining when to stop
    max_iter: Maximum number of iterations of the KMeans
    centroids: save the result (not sure if necessary)

    the default settings are mostly adapted from KMeans class in sklearn
    '''
    def __init__(self, K=8, seed=42, tol=0.5, max_iter=30):
        self.K = K
        self.seed = seed
        self.tol = tol
        self.max_iter = max_iter

    def euclidean_distance(self, pt_a, centroid):
        pairwise_dist = [(float(a)-float(b))**2 for a,b in zip(pt_a[1], centroid)]
        eucli_dist = math.sqrt(sum(pairwise_dist))
        #print(eucli_dist)
        return eucli_dist

    # initialize centroids
    # randomly choose some data points in the set as initial centroids
    def init_centroids(self, data):
        random.seed(self.seed)
        init_K = self.K
        try:
            random_lst = random.sample(data, init_K)
        except:
            random_lst = random.choices(data, k=init_K)
        # only return coordinates
        init_centroid_lst = [data[1] for data in random_lst]
        return init_centroid_lst

    # find the closest centroid to a point
    def find_closest_centroid(self, point, centroids):
        min_dist = sys.maxsize
        closest_c = None
        for centroid in centroids:
            dist = self.euclidean_distance(point, centroid)
            #print(dist)
            if dist < min_dist:
                closest_c = centroid
                min_dist = dist
        return closest_c

    def update_centroids(self, centroids, X_data, y_label, l_to_c):
        # add up all data points with same labelling
        # divided by number of same label points
        # calculate distance moved, compare with tol, determine stop or not
        label_result = dict()
        label_count = dict()
        for data, label in zip(X_data, y_label):
            if not label_result.get(label):
                label_result[label] = data[1]
                label_count[label] = 1
            else:
                label_result[label] = [float(a)+float(b) for a,b in zip(label_result[label], data[1])]
                label_count[label] += 1
        new_l_to_c = dict()
        new_c_to_l = dict()
        new_centroids = []
        total_move = 0
        for label, sum in label_result.items():
            count = label_count.get(label)
            new_centroid = [str(float(a)/count) for a in sum]
            new_l_to_c[label] = new_centroid
            new_c_to_l[tuple(new_centroid)] = label
            new_centroids.append(new_centroid)
            old_centroid = l_to_c[label]
            total_move += self.euclidean_distance(['dummy', new_centroid], old_centroid)

        stop = False
        #print(total_move)
        if total_move < self.tol * self.K:
            stop = True
        return new_centroids, new_l_to_c, new_c_to_l, stop

    def fit(self, X):
        centroids = self.init_centroids(X)
        #print(centroids)
        # labelling the centroids
        label_centroids = OrderedDict() # just for testing convenience, normal dict is fine
        centroids_label = dict()
        for index, centroid in enumerate(centroids):
            label_centroids[str(index)] = centroid
            centroids_label[tuple(centroid)] = str(index)
        #print(centroids_label)
        # init label results [-1,-1,-1,....]
        y_label = len(X) * [-1]
        i=0
        while i < self.max_iter:
        #while i<1:
            for j in range(0, len(X)):
                # closest returns coordinates (a list)
                closest = self.find_closest_centroid(X[j], centroids)
                # make sure label result should have same index as X
                y_label[j] = centroids_label[tuple(closest)]

            centroids, label_centroids, centroids_label, enough = self.update_centroids(centroids, X, y_label, label_centroids)
            if enough:
                break
            #print(i)
            i += 1

        return y_label

def zipping(X_data, y_label):
    X_dict = dict()
    y_X_dict = dict()
    X_dict = {a[0]: a[1] for a in X_data}
    for data, label in zip(X_data, y_label):
        if not y_X_dict.get(label):
            y_X_dict[label] = [data[0]]
        else:
            y_X_dict[label].append(data[0])
    return X_dict, y_X_dict

def update_cluster(res, y_X_dict):
    for label, points in y_X_dict.items():
        if not res.get(label):
            res[label] = points
        else:
            res[label].extend(points)
    return res

def update_CS(curr, X_dict, y_X_dict):
    for label, points in y_X_dict.items():
        # {label: [[index, coordinates]]}
        for point in points:
            if not curr.get(label):
                curr[label] = [[point, X_dict[point]]]
            else:
                curr[label].append([point, X_dict[point]])
    return curr

def init_DS_stats(X_dict, y_X_dict):
    N = dict()
    SUM = dict()
    SUMSQ = dict()
    for cluster_k, points in y_X_dict.items():
        N[cluster_k] = len(points)
        for index in points:
            if not SUM.get(cluster_k):
                SUM[cluster_k] = X_dict[index]
                SUMSQ[cluster_k] = [float(a)**2 for a in X_dict[index]]
            else:
                SUM[cluster_k] = [float(a)+float(b) for a,b in zip(SUM[cluster_k], X_dict[index])]
                SUMSQ[cluster_k] = [float(a)+float(b)**2 for a,b in zip(SUMSQ[cluster_k], X_dict[index])]
    return N, SUM, SUMSQ

def init_CS_stats(X_dict, y_X_dict):
    N = dict()
    SUM = dict()
    SUMSQ = dict()
    RS = []
    new_y_X_dict = dict()
    for cluster_k, points in y_X_dict.items():
        if len(points)>1:
            new_y_X_dict[cluster_k] = points
            N[cluster_k] = len(points)
            for index in points:
                if not SUM.get(cluster_k):
                    SUM[cluster_k] = X_dict[index]
                    SUMSQ[cluster_k] = [float(a)**2 for a in X_dict[index]]
                else:
                    SUM[cluster_k] = [float(a)+float(b) for a,b in zip(SUM[cluster_k], X_dict[index])]
                    SUMSQ[cluster_k] = [float(a)+float(b)**2 for a,b in zip(SUMSQ[cluster_k], X_dict[index])]
        else:
            for index in points:
                RS.append([index, X_dict[index]])
    return N, SUM, SUMSQ, RS, new_y_X_dict

def mahalanobis_distance(point, N, S, SQ):
    #std = [math.sqrt((sq/N)-((s/N)**2)) for s, sq in zip(S, SQ)]
    # mah dist: sqrt(sum((x-y)^2/std^2))
    try:
        pairwise_dist = [((float(pt)-s/N)**2)/((sq/N)-((s/N)**2)) for pt, s, sq in zip(point, S, SQ)]
        mahala_dist = math.sqrt(sum(pairwise_dist))
    except:
        mahala_dist = sys.maxsize
    return mahala_dist

def find_nearest_cluster(data, clusters, N, S, SQ):
    min_dist = sys.maxsize
    nearest = '-1'
    for cluster in clusters:
        dist = mahalanobis_distance(data, N[cluster], S[cluster], SQ[cluster])
        if dist<min_dist:
            min_dist = dist
            nearest = cluster
    return nearest, min_dist

def update_stats(point, cluster_k, N, S, SQ):
    N[cluster_k] += 1
    S[cluster_k] = [float(a)+float(b) for a,b in zip(S[cluster_k], point)]
    SQ[cluster_k] = [float(a)+float(b)**2 for a,b in zip(SQ[cluster_k], point)]
    return N, S, SQ

def add_CS(CS_dict, CS_N, CS_SUM, CS_SUMSQ, new_CS_N, new_CS_SUM, new_CS_SUMSQ, X_dict, y_X_dict):
    new_keys = new_CS_N.keys()
    keys = CS_dict.keys()
    for key in new_keys:
        # if exists, create a new one and add
        if CS_dict.get(key):
            old_key = key
            while True:
                new_key = str(random.randint(0, 10000))
                if new_key not in keys:
                    break
            for point in y_X_dict[old_key]:
                if not CS_dict.get(new_key):
                    CS_dict[new_key] = [[point, X_dict[point]]]
                else:
                    CS_dict[new_key].append([point, X_dict[point]])
            CS_N[new_key] = new_CS_N[old_key]
            CS_SUM[new_key] = new_CS_SUM[old_key]
            CS_SUMSQ[new_key] = new_CS_SUMSQ[old_key]
        else:
            for point in y_X_dict[key]:
                if not CS_dict.get(key):
                    CS_dict[key] = [[point, X_dict[point]]]
                else:
                    CS_dict[key].append([point, X_dict[point]])
            CS_N[key] = new_CS_N[key]
            CS_SUM[key] = new_CS_SUM[key]
            CS_SUMSQ[key] = new_CS_SUMSQ[key]
    return CS_dict, CS_N, CS_SUM, CS_SUMSQ

def merge_CS(CS_dict, N, S, SQ, d):
    if not CS_dict:
        return CS_dict, N, S, SQ
    curr_labels  = CS_dict.keys()
    merged_labels = []
    new_labels = []
    #print(curr_labels)
    #test = []
    while True:
        merge_count = 0
        for pairs in combinations(curr_labels, 2):
            # if not already merged
            if pairs[0] in merged_labels or pairs[1] in merged_labels:
                continue
            # calculate dist between two clusters
            countN = N[pairs[0]]
            center_0 = [sum/countN for sum in S[pairs[0]]]
            dist = mahalanobis_distance(center_0, N[pairs[1]], S[pairs[1]], SQ[pairs[1]])
            # if < 2sqrt(d) add;remove;count
            if dist < alpha*math.sqrt(d):
                while True:
                    new_label = str(random.randint(0, 10000))
                    if new_label not in curr_labels and new_label not in new_labels:
                        break
                #test.append(str(pairs[0]+'+'+pairs[1]+'='+new_label))
                CS_dict[pairs[0]].extend(CS_dict[pairs[1]])
                CS_dict[new_label] = CS_dict[pairs[0]]
                new_N = N[pairs[0]]+N[pairs[1]]
                N[new_label] = new_N
                new_sum = [a+b for a,b in zip(S[pairs[0]], S[pairs[1]])]
                S[new_label] = new_sum
                new_sq = [a+b for a,b in zip(SQ[pairs[0]], SQ[pairs[1]])]
                SQ[new_label] = new_sq
                del CS_dict[pairs[0]]
                del CS_dict[pairs[1]]
                del N[pairs[0]]
                del N[pairs[1]]
                del S[pairs[0]]
                del S[pairs[1]]
                del SQ[pairs[0]]
                del SQ[pairs[1]]
                merged_labels.append(pairs[0])
                merged_labels.append(pairs[1])
                new_labels.append(new_label)
                merge_count += 1
        for ele in curr_labels:
            if ele not in merged_labels and ele not in new_labels:
                new_labels.append(ele)
        curr_labels = new_labels
        new_labels = []
        merged_labels = []
        if merge_count==0:
            break

    return CS_dict, N, S, SQ

def add_CS_to_DS(CS_dict, CS_key, cluster, res, N, SUM, SUMSQ):
    for data in CS_dict[CS_key]:
        res = update_cluster(res, {cluster: [data[0]]})
        N, SUM, SUMSQ = update_stats(data[1], nearest, N, SUM, SUMSQ)

    return res, N, SUM, SUMSQ

if __name__ == '__main__':
    start = time.time()

    # input
    input_path = sys.argv[1]
    cluster_num = int(sys.argv[2])
    output_final = sys.argv[3]
    output_intermediate = sys.argv[4]

    # load each chunk of data
    # need to check if first chunk
    data_chunks = glob.glob(input_path+'/*.txt')
    file_num = len(data_chunks)
    for count, chunk in enumerate(data_chunks):
        with open(chunk, 'r') as f:
            lines = f.readlines()
            data_list = [point.strip().split(',') for point in lines]
            data_points = [[point[0], list(map(lambda x: x.encode('utf-8'), point[1:]))] for point in data_list]
            # # of data in this file
            data_num = len(data_points)
            # dimension
            d = len(data_points[0][1])
            if d < 50 and data_num < 50000:
                alpha = 1.5
            else:
                alpha = 2.0
        # if the first file
        if count == 0:
            # get a small portion of data for initialization (1/10)
            init_X = data_points[:math.floor(0.1*data_num)]
            # run KMeans with 3 times K
            y_label = KMeans(K=3*cluster_num, tol=0.05*d).fit(init_X)
            # create {index: coordinates} dict for searching
            # zip X index with y_label, groupby labels
            X_dict, y_X_dict = zipping(init_X, y_label)
            # initialize RS: []
            RS = []
            inlier_points = []
            for label, indices in y_X_dict.items():
                # if very few points in the cluster, then -> RS, else -> inlier points
                if len(indices) < len(init_X)*0.02:
                    for index in indices:
                        RS.append([index, X_dict[index]])
                else:
                    for index in indices:
                        inlier_points.append([index, X_dict[index]])
            # run KMeans on inlier points, assign index to cluster, initialze DS stats: {}
            cluster_res = dict()
            y_label = KMeans(K=cluster_num, tol=0.05*d).fit(inlier_points)
            X_dict, y_X_dict = zipping(inlier_points, y_label)
            cluster_res = update_cluster(cluster_res, y_X_dict)
            DS_N, DS_SUM, DS_SUMSQ = init_DS_stats(X_dict, y_X_dict)

            #print("are you here?")
            #print(RS)
            # run KMeans on current RS (outliers) with 3*K, initialize CS stats: {}
            if not RS:
                CS_dict = dict()
                CS_N = dict()
                CS_SUM = dict()
                CS_SUMSQ = dict()
            else:
                CS_dict = dict()
                y_label = KMeans(K=3*cluster_num, tol=0.05*d).fit(RS)
                #print("and here?")
                X_dict, y_X_dict = zipping(RS, y_label)
                CS_N, CS_SUM, CS_SUMSQ, RS, y_X_dict = init_CS_stats(X_dict, y_X_dict)
                CS_dict = update_CS(CS_dict, X_dict, y_X_dict)

            # load remaining file
            X = data_points[math.floor(0.1*data_num):]
            # for each data point, find nearest DS using mahalanobis distance
            for data in data_points:
                clusters = DS_N.keys()
                nearest, dist = find_nearest_cluster(data[1], clusters, DS_N, DS_SUM, DS_SUMSQ)
                # if < alpha*sqrt(dimension) (let alpha = 2), move to this DS cluster
                # add to cluster and update DS stats
                if dist<alpha*math.sqrt(d):
                    cluster_res = update_cluster(cluster_res, {nearest: [data[0]]})
                    DS_N, DS_SUM, DS_SUMSQ = update_stats(data[1], nearest, DS_N, DS_SUM, DS_SUMSQ)
                    continue
                # else find nearest CS cluster
                clusters = CS_N.keys()
                nearest, dist = find_nearest_cluster(data[1], clusters, CS_N, CS_SUM, CS_SUMSQ)
                # if < alpha*sqrt(dimension) move to CS
                # add to CS and update stats
                if dist<alpha*math.sqrt(d):
                    CS_dict = update_CS(CS_dict, {data[0]: data[1]}, {nearest: [data[0]]})
                    CS_N, CS_SUM, CS_SUMSQ = update_stats(data[1], nearest, CS_N, CS_SUM, CS_SUMSQ)
                    continue
                # else add data to RS
                RS.append([data[0], data[1]])
            # run KMeans on RS with 3*K (if large enough)
            # add new result to CS
            if len(RS)>=3*cluster_num:
                y_label = KMeans(K=3*cluster_num, tol=0.05*d).fit(RS)
                X_dict, y_X_dict = zipping(RS, y_label)
                new_CS_N, new_CS_SUM, new_CS_SUMSQ, RS, y_X_dict = init_CS_stats(X_dict, y_X_dict)
                CS_dict, CS_N, CS_SUM, CS_SUMSQ = add_CS(CS_dict, CS_N, CS_SUM, CS_SUMSQ, new_CS_N, new_CS_SUM, new_CS_SUMSQ, X_dict, y_X_dict)
            # compare CS, merge if dist<2*sqrt(d)
            CS_dict, CS_N, CS_SUM, CS_SUMSQ = merge_CS(CS_dict, CS_N, CS_SUM, CS_SUMSQ, d)

            # output intermediate results
            # ref: https://www.geeksforgeeks.org/writing-csv-files-in-python/
            fields = ['round_id', 'nof_cluster_discard', 'nof_point_discard', 'nof_cluster_compression', 'nof_point_compression', 'nof_point_retained']
            round_id = count+1
            nof_cluster_discard = len(DS_N.keys())
            nof_point_discard = 0
            for num in DS_N.values():
                nof_point_discard += num
            nof_cluster_compression = len(CS_N.keys())
            nof_point_compression = 0
            for num in CS_N.values():
                nof_point_compression += num
            nof_point_retained = len(RS)
            rows = []
            row = [round_id, nof_cluster_discard, nof_point_discard, nof_cluster_compression, nof_point_compression, nof_point_retained]
            rows.append(row)
        else:
            # load remaining file
            X = data_points
            # for each data point, find nearest DS using mahalanobis distance
            for data in data_points:
                clusters = DS_N.keys()
                nearest, dist = find_nearest_cluster(data[1], clusters, DS_N, DS_SUM, DS_SUMSQ)
                # if < alpha*sqrt(dimension) (let alpha = 2), move to this DS cluster
                # add to cluster and update DS stats
                if dist<alpha*math.sqrt(d):
                    cluster_res = update_cluster(cluster_res, {nearest: [data[0]]})
                    DS_N, DS_SUM, DS_SUMSQ = update_stats(data[1], nearest, DS_N, DS_SUM, DS_SUMSQ)
                    continue
                # else find nearest CS cluster
                clusters = CS_N.keys()
                nearest, dist = find_nearest_cluster(data[1], clusters, CS_N, CS_SUM, CS_SUMSQ)
                # if < alpha*sqrt(dimension) move to CS
                # add to CS and update stats
                if dist<alpha*math.sqrt(d):
                    CS_dict = update_CS(CS_dict, {data[0]: data[1]}, {nearest: [data[0]]})
                    CS_N, CS_SUM, CS_SUMSQ = update_stats(data[1], nearest, CS_N, CS_SUM, CS_SUMSQ)
                    continue
                # else add data to RS
                RS.append([data[0], data[1]])
            # run KMeans on RS with 3*K (if large enough)
            # add new result to CS
            if len(RS)>=3*cluster_num:
                y_label = KMeans(K=3*cluster_num, tol=0.05*d).fit(RS)
                X_dict, y_X_dict = zipping(RS, y_label)
                new_CS_N, new_CS_SUM, new_CS_SUMSQ, RS, y_X_dict = init_CS_stats(X_dict, y_X_dict)
                CS_dict, CS_N, CS_SUM, CS_SUMSQ = add_CS(CS_dict, CS_N, CS_SUM, CS_SUMSQ, new_CS_N, new_CS_SUM, new_CS_SUMSQ, X_dict, y_X_dict)
            # compare CS, merge if dist<2*sqrt(d)
            CS_dict, CS_N, CS_SUM, CS_SUMSQ = merge_CS(CS_dict, CS_N, CS_SUM, CS_SUMSQ, d)

            # if last run merge all CS to nearest DS, merge RS to nearest DS
            if count == file_num-1:
                for data in RS:
                    clusters = DS_N.keys()
                    nearest, dist = find_nearest_cluster(data[1], clusters, DS_N, DS_SUM, DS_SUMSQ)
                    cluster_res = update_cluster(cluster_res, {nearest: [data[0]]})
                    DS_N, DS_SUM, DS_SUMSQ = update_stats(data[1], nearest, DS_N, DS_SUM, DS_SUMSQ)
                RS = []

                # calculate center of CS as a point find neearest cluster
                # update everything to cluster_res; clear CS stuff
                for key in CS_dict.keys():
                    clusters = DS_N.keys()
                    countN = CS_N[key]
                    centroid = [sum/countN for sum in CS_SUM[key]]
                    nearest, dist = find_nearest_cluster(centroid, clusters, DS_N, DS_SUM, DS_SUMSQ)
                    cluster_res, DS_N, DS_SUM, DS_SUMSQ = add_CS_to_DS(CS_dict, key, nearest, cluster_res, DS_N, DS_SUM, DS_SUMSQ)
                CS_N=dict()
                CS_SUM=dict()
                CS_SUMSQ=dict()

            # save intermediate results
            round_id = count+1
            nof_cluster_discard = len(cluster_res.keys())
            nof_point_discard = 0
            for num in cluster_res.values():
                nof_point_discard += len(num)
            nof_cluster_compression = len(CS_N.keys())
            nof_point_compression = 0
            for num in CS_N.values():
                nof_point_compression += num
            nof_point_retained = len(RS)
            row = [round_id, nof_cluster_discard, nof_point_discard, nof_cluster_compression, nof_point_compression, nof_point_retained]
            rows.append(row)

        with open(output_intermediate, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(fields)
            csvwriter.writerows(rows)

    # generate result dict
    res_dict = dict()
    for label, points in cluster_res.items():
        for index in points:
            res_dict[index] = int(label)

    with open(output_final, 'w') as jsonfile:
        jsonfile.writelines(json.dumps(res_dict))

    end = time.time()
    print("Duration: "+str(end-start))
