from sklearn.metrics import normalized_mutual_info_score
import sys
import json


def main():

    true_label_path = sys.argv[1]
    predict_label_path = sys.argv[2]

    with open(true_label_path, "r") as f:
        true_label_dict = json.load(f)

    with open(predict_label_path, "r") as f:
        predict_label_dict = json.load(f)

    true_label_ls = [-1] * len(true_label_dict)
    for point_id, cluster_id in true_label_dict.items():
        true_label_ls[int(point_id)] = cluster_id

    predict_label_ls = [-1] * len(predict_label_dict)
    for point_id, cluster_id in predict_label_dict.items():
        predict_label_ls[int(point_id)] = cluster_id

    NMI = normalized_mutual_info_score(true_label_ls, predict_label_ls)
    print("NMI: %.5f" % NMI)


if __name__ == "__main__":
    main()
