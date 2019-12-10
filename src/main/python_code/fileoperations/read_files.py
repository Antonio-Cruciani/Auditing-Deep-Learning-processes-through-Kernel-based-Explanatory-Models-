import gzip
import json

def get_landmarks_from_file(landmarks_path):
    list_of_landmarks = []
    with gzip.open(landmarks_path + "landmarks.txt.gz", 'rt') as f:
        landmarks_file = json.load(f)
        for k in landmarks_file:
            list_of_landmarks.append([k["label"], k["question"]])

    return list_of_landmarks

def get_labels_from_file(labels_path):
    label_map={}
    label_inverted_map = {}
    with open(labels_path+"classes_labels.txt", 'rt') as f:
        i = 0
        for line in f:
            label_inverted_map[line.split("\n")[0]] = i
            label_map[i] = line.split("\n")[0]
            i += 1
    return label_map,label_inverted_map