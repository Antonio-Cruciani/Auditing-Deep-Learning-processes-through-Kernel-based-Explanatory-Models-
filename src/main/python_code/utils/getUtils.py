import gzip

def getLabels(fileDirs):
    labels_to_int_map = dict()
    i = 0
    for filePath in fileDirs:
        with gzip.open(filePath, 'rt') as f:
            for line in f:
                split = line.split("\t")
                label = split[1].strip("\n")
                if ((label in labels_to_int_map) == False):
                    labels_to_int_map[label] = i
                    i += 1
                    # print("Key=",label," Value=",labelsToIntMap[label])
        #print("Added ", len(labels_to_int_map.keys()), " labels")

    return labels_to_int_map

def getInvertedMap(dictionary):
    inverted_dict = dict()
    for word in dictionary:
        inverted_dict[dictionary[word]]= word

    return inverted_dict


