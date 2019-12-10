import tensorflow as tf


class NyProjector:
    """ This class is responsible of applying the Nystrom projection, i.e., it multiplies a given an input vector
        (the C in the Nystrom method) with the projection matrix H_Ny (unless a fake projection is required,
        which means that the input vector is not multiplied)
    """

    def __init__(self, projection_matrix):
        self.projection_matrix = projection_matrix.get_projection_matrix()  # TensorFlow Variable
        self.projection_size = projection_matrix.get_columns_size()
        #self.embedding_matrix = embedding_matrix                            # Embedding Matrix definizione? Da dove arriva? ==> Tensore o Lista di tensori di stessa dimensione

    def project(self, input_vec, dkp=1.0, fake_projection=False):
        # Load the C vectors from the "embedding"
        """

        :param instance_index: index of the c vector in the Nystrom formulation
        :param dkp: drop keep probability
        :param fake_projection: whether the c vector must be multiplied for the Nystrom projection matrix
        :return: the c vector projected in the Nystrom space ( x_tilde of the paper)
        """
        #with tf.device('/cpu:0') and tf.name_scope("c_embedding"):
            #input_vec = tf.squeeze(tf.nn.embedding_lookup(self.embedding_matrix, instance_index, name="x_vec"),   # This function is used to perform parallel lookups on the list of tensors in params, single tensor representing the complete
            #                        squeeze_dims=[1])                                                              # embedding tensor, or a list of P tensors all of same shape except for the first dimension, representing sharded embedding tensors.

        #    input_vec = tf.nn.dropout(input_vec, dkp)

        # Project x1 and x2 through the projection matrix
        with tf.name_scope("projection"):
            if fake_projection:
                return tf.nn.dropout(input_vec, dkp)   # Per quale motivo fare dropout? dkp is a scalar Tensor with the same type as input_vec. Probability that each element is kept
            else:
                return tf.nn.dropout(tf.matmul(input_vec, self.projection_matrix, name="x_projection"), dkp)     # name = "nome_della_operazione" (input opzionale)
                                                                                                                 # Restituisce un tensore delle stesse dimensioni di input_vec
