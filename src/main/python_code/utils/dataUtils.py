import gzip
import numpy as np
from src.main.python_code.data.instance import Instance


# This function load a dataset as a list of Instance objects (see instance.py)
def loadData(filePath):
    data = []
    with gzip.open(filePath, 'rt') as f:
        for line in f:
            split = line.split("\t")
            instance_id = int(split[0])
            string_label = str(split[1].strip())
            data.append(Instance(instance_id,string_label))
    return data

# This function split dataset according to splitting parameter split_f
def splitData(dataset, split_f):
    np.random.shuffle(dataset)  # np.random.shuffle() cambia direttamente dataset
    split_size = int(len(dataset) * split_f)
    return dataset[0:split_size], dataset[split_size:]

# This function maps the label of each instance into a one-hot vector representation according to label map
# See instance.py for further details
def get_arrays_from(dataset, label_map):
    x = [np.asarray([p.id]) for p in dataset]            # lista di numpy.ndarray costituite da un solo int32
    y = [p.get_vector_label(label_map) for p in dataset] # lista di array di float64 lunghe quanto label_map e costituite da un 1.0 e tutti 0.0
    return np.array(x), np.array(y)                      # Restituisce una numpy.ndarray di numpy.ndarray
