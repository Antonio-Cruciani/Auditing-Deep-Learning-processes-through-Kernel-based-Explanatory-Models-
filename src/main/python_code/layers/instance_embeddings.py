import tensorflow as tf
import gzip
import numpy as np
import sys


class InstanceEmbedding:
    """ This class is used to store instance embeddings, i.e., dense vectors associated to an instance.
    The procedure for generating the Embeddings is completely transparent for this class. For instance, it can be used
    to store the C vectors of the Nystrom formulation

    """

    def __init__(self, matrix_path, train_embeddings):
        """ Constructor

        :param matrix_path: path of the embeddings file
        :param train_embeddings: whether the embeddings are trainable
        """
        self.train_embeddings = train_embeddings
        self.labels = []
        self.W = None
        self.W_np_format = None
        # with gzip.open(matrix_path, 'rb') as f:
        with gzip.open(matrix_path, 'rt') as f:
            i = 0
            for line in f:
                if i > 0:
                    sp = line.split("\t")
                    # term = sp[0]
                    self.labels.append(sp[1])
                    vector = sp[3].split(",")
                    for j, value in enumerate(vector):
                        self.W_np_format[i - 1][j] = value
                else:  # first row
                    self.rows, self.columns = line.split()
                    self.rows = int(self.rows)
                    self.columns = int(self.columns)
                    #print("Loading rows {} of the embedding with columns {}".format(self.rows, self.columns))
                    # self.W_np_format = np.random.uniform(-1.0, 1.0, size=(self.rows, self.columns)).astype("float32")
                    self.W_np_format = np.zeros((self.rows, self.columns)).astype('float32')
                i += 1

            with tf.device('/cpu:0'), tf.name_scope("embedding"):
                self.W = tf.Variable(self.W_np_format, name="EmbeddingNoVoc", trainable=self.train_embeddings)

    def get_rows(self):
        return self.rows

    def get_embedding_size(self):
        return self.columns

    def get_W(self):
        return self.W

    def get_numpy_w(self):
        return self.W_np_format

    def get_labels(self):
        return self.labels
