import tensorflow as tf
import gzip
import numpy as np


class ProjectionMatrix:
    """ This class corresponds to a Nystrom Projection Matrix (in  pratica H_Ny (U x sqrt(S)) i cui pesi posso considerarli fissi in fase
         di training (ottenuti dalla SVD) oppure coinvolgerli nella BackPropagation))
    """

    def __init__(self, matrix_path, trainable=False):
        """ Constructor

        :param matrix_path: path of the Nystrom projection matrix  ( presumo di conoscerla )
        :param trainable: whether the back-propagation should modify the projection matrix (Booleano che fisso inizialmente come falso )
        """
        self.matrix_path = matrix_path
        self.trainable = trainable
        self.__load_matrix()

    def __load_matrix(self):
        # Questa funzione carica la matrice di proiezione di Nystrom da un file esterno di cui ho dato il path come input e la salva
        # come un Variable in TensorFlow W e una array numpy wTmp di cui salvo le dimensioni (numero di righe e numero di colonne)

        self.W = None
        # with gzip.open(self.matrix_path, 'rb') as f:
        with gzip.open(self.matrix_path, 'rt') as f:
            i = 0             # Deve esserci un caso particolare nella prima riga
            for line in f:
                if i > 0:
                    vector = line.split(",")
                    for j, value in enumerate(vector):
                        self.wTmp[i - 1][j] = value
                else:
                    self.rows, self.columns = line.split()
                    self.rows = int(self.rows)         # deve essere un intero in quanto int() prende come input solo una stringa, non una lista di stringhe
                    self.columns = int(self.columns)
                    #print("Loading rows {} of the projection matrix with columns {}".format(self.rows, self.columns))
                    self.wTmp = np.random.uniform(-1.0, 1.0, size=(self.rows, self.columns)).astype("float32") # inizializzo l'array numpy wTmp
                i += 1

            with tf.device('/cpu:0'), tf.name_scope("projection_matrix"):     # tf.name_scope("stringa")  = Initialize the context manager for the computational graph
                self.W = tf.Variable(self.wTmp, name="ProjectionMatrix", trainable=self.trainable)

    def get_projection_matrix(self):
        # Restituisce la matrice di proiezione come un oggetto di TensorFlow
        return self.W

    def get_numpy_w(self):
        # Restituisce la matrice di proiezione come una array numpy
        return self.wTmp

    def get_rows_size(self):
        # Restituisce il numero di righe
        return self.rows

    def get_columns_size(self):
        # Restituisce il numero di colonne
        return self.columns
