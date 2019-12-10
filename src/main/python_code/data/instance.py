import numpy as np


class Instance:
    """ This class represents a classification example, i.e., a pair <id, label> where
     id is the index of the example, and label is its classification label
    """
    #def __init__(self, instance_id, string_label):
    def __init__(self, instance_vector, string_label):
        self.vector = instance_vector                   # istance vector
        #self.id = instance_id                          # instance_id deve essere un "int"
        self.string_label = string_label               # string_label in genere e' una "str"
    # Return the field vector that is a np.array
    def get_vector(self):
        return self.vector


    def get_vector_label(self, label_map):
        """

        :param label_map: a mapping from a textual format to a corresponding integer value
        :return: a one-hot vector where all the entries are 0 except a 1 in the position of the example label (which is
        derived from label_map).
        """
        y = np.zeros(len(label_map))         # label_map e' un "dict"
        y[label_map[self.string_label]] = 1
        return y
