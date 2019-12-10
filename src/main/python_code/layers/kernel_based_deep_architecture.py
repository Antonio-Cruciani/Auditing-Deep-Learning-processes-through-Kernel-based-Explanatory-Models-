import tensorflow as tf
import sys
from src.main.python_code.utils.nnUtils import activate, softmax_cross_entropy # Activation function for neurons and Loss Function
import tensorflow as tf

def simple_lrp_linear(R, input_tensor, weights, biases=None):
    R_shape = R.get_shape().as_list()
    if len(R_shape)!=2:
        linear = tf.nn.bias_add(tf.matmul(input_tensor, weights), biases)
        activations_shape = linear.get_shape().as_list()
        # activations_shape = self.activations.get_shape().as_list()
        R = tf.reshape(R, activations_shape)

    Z = tf.expand_dims(weights, 0) * tf.expand_dims(input_tensor, -1)
    Zs = tf.expand_dims(tf.reduce_sum(Z, 1), 1)
    if biases:
         Zs += tf.expand_dims(tf.expand_dims(biases, 0), 0)
    stabilizer = 1e-8*(tf.where(tf.greater_equal(Zs,0), tf.ones_like(Zs, dtype=tf.float32), tf.ones_like(Zs, dtype=tf.float32)*-1))
    Zs += stabilizer

    return tf.reduce_sum((Z / Zs) * tf.expand_dims(R, 1),2)

    # def _simple_lrp(self, R):  # R sarebbe il relevance score del Layer considerato
    #     self.R = R
    #     # Controllo se il tensore degli score di Relevance ha le stesse dimensioni del tensore di attivazioni (num. di neuroni)
    #     R_shape = self.R.get_shape().as_list()
    #     if len(R_shape) != 2:
    #         activations_shape = self.activations.get_shape().as_list()
    #         self.R = tf.reshape(self.R, activations_shape)
    #
    #     Z = tf.expand_dims(self.weights, 0) * tf.expand_dims(self.input_tensor,
    #                                                          -1)  # tf.expand_dims aggiunge una dimensione al tensore desiderato (e.g. 0 all'inizio e -1 alla fine)
    #     Zs = tf.expand_dims(tf.reduce_sum(Z, 1), 1) + tf.expand_dims(tf.expand_dims(self.biases, 0), 0)
    #     stabilizer = 1e-8 * (
    #     tf.where(tf.greater_equal(Zs, 0), tf.ones_like(Zs, dtype=tf.float32), tf.ones_like(Zs, dtype=tf.float32) * -1))
    #     Zs += stabilizer
    #
    #     return tf.reduce_sum((Z / Zs) * tf.expand_dims(self.R, 1),
    #                          2)  # Restituisce i relevance score del Layer precedente (dipendono dai pesi W ma anche dagli attuali score di R)


def epsilon_lrp(R, epsilon, input_tensor, weights, biases=None):
    '''
    LRP according to Eq(58) in DOI: 10.1371/journal.pone.0130140
    '''
    Z = tf.expand_dims(weights, 0) * tf.expand_dims(input_tensor, -1)
    Zs = tf.expand_dims(tf.reduce_sum(Z, 1), 1)
    if biases:
        Zs += tf.expand_dims(tf.expand_dims(biases, 0), 0)
    Zs += epsilon * tf.where(tf.greater_equal(Zs,0), tf.ones_like(Zs)*-1, tf.ones_like(Zs))

    return tf.reduce_sum((Z / Zs) * tf.expand_dims(R, 1),2)

    # def _alphabeta_lrp(self,R,alpha):
    #     '''
    #     LRP according to Eq(60) in DOI: 10.1371/journal.pone.0130140
    #     '''
    #     self.R= R
    #     beta = 1 - alpha
    #     Z = tf.expand_dims(self.weights, 0) * tf.expand_dims(self.input_tensor, -1)
#
#     if not alpha == 0:
#         Zp = tf.where(tf.greater(Z,0),Z, tf.zeros_like(Z))
#         term2 = tf.expand_dims(tf.expand_dims(tf.where(tf.greater(self.biases,0),self.biases, tf.zeros_like(self.biases)), 0 ), 0)
#         term1 = tf.expand_dims( tf.reduce_sum(Zp, 1), 1)
#         Zsp = term1 + term2
#         Ralpha = alpha * tf.reduce_sum((Zp / Zsp) * tf.expand_dims(self.R, 1),2)
#     else:
#         Ralpha = 0
#
#     if not beta == 0:
#         Zn = tf.where(tf.less(Z,0),Z, tf.zeros_like(Z))
#         term2 = tf.expand_dims(tf.expand_dims(tf.where(tf.less(self.biases,0),self.biases, tf.zeros_like(self.biases)), 0 ), 0)
#         term1 = tf.expand_dims( tf.reduce_sum(Zn, 1), 1)
#         Zsp = term1 + term2
#         Rbeta = beta * tf.reduce_sum((Zn / Zsp) * tf.expand_dims(self.R, 1),2)
#     else:
#         Rbeta = 0
#
#     return Ralpha + Rbeta


def simple_lrp_softmax(R, input_tensor, *args, **kwargs):
    # component-wise operations within this layer
    #Rx = self.input_tensor  * self.activations
    Rx = input_tensor * R
    #Rx = Rx / tf.reduce_sum(self.input_tensor)
    return Rx

class KernelBasedDeepArchitecture:
    """ This class implements the Kernel-based Deep Architecture (KDA) described in:
    Croce D., Filice S., Castellucci G. and Basili R. Deep Learning in Semantic Kernel Spaces. In Proceedings of ACL '17

    """

    def __init__(self, nn_typeA, projector_s, projector_d, hl_conf_list, l2_lambda, num_classes,landmarks_size, rnd_seed=1927):
        self.type = nn_typeA
        self.ny_projector_s = projector_s     # Proiettore (c x H_Ny) Statico
        self.ny_projector_d = projector_d     # Proiettore (c x H_Ny) Dinamico
        self.hl_list = hl_conf_list           # Lista con numero di HiddenLayerConf (classe definita in hidden_layer_configuration)
        self.num_classes = num_classes
        self.l2_lambda = l2_lambda
        self.rnd_seed = rnd_seed
        self.landmarks_size = landmarks_size

        self.input_x = tf.placeholder(tf.float32, shape=[None, self.landmarks_size], name="input_x")  # the KDA "x" input is actually
        # the index of the C vector to be used

        self.input_y = tf.placeholder(tf.float32, shape=[None, self.num_classes], name="input_y")  # the KDA "y" input
        # (necessary in training) is a "one-hot vector" whose dimensionality corresponds to the number of classes

        self.dkp = tf.placeholder(tf.float32, name="dropout_keep_prob")     # A scalar Tensor of float32 type

        in_size = self.ny_projector_s.projection_size     # Numero di colonne della matrice H_Ny o meglio "l" numero di landmarks

        self.l2_loss = 0

        # LRP:
        self.is_test = tf.placeholder(tf.float32, name="is_test")
        self.LRP_W_s = []
        self.LRP_W_d = []
        self.LRP_b_s = []
        self.LRP_b_d = []
        self.LRP_input_tensors_s = []
        self.LRP_input_tensors_d = []

        # PRIMO LAYER (NYSTROM LAYER): PROIEZIONE DEL VETTORE DI INPUT "x":
        with tf.name_scope("x_projection"):  # "x_projection": guarda ny_projector.project(instance_index, dkp=1.0, fake_projection=False)

            if self.type == "n":
                self.projection_s = self.ny_projector_s.project(self.input_x, fake_projection=True) # Proiezione espressa come tensore
                self.hl_projection_s = self.projection_s

            if self.type == "s" or self.type == "sd":
                self.projection_s = self.ny_projector_s.project(self.input_x)
                self.hl_projection_s = self.projection_s

                # Ponendo fake_projection=True, ny_projector_s.project(instance_index) restituisce lo stesso embedding "c"
                self.LRP_input_tensors_s.append(self.ny_projector_s.project(self.input_x, dkp=1.0, fake_projection=True))
                # La prima matrice nella lista e' la matrice di proiezione di Nystrom
                self.LRP_W_s.append(self.ny_projector_s.projection_matrix)
                # Nel Nystrom Layer non ho bias
                self.LRP_b_s.append(None)

            if self.type == "d" or self.type == "sd":
                self.projection_d = self.ny_projector_d.project(self.input_x)
                self.hl_projection_d = self.projection_d
                self.LRP_input_tensors_d.append(self.ny_projector_d.project(self.input_x, dkp=1.0, fake_projection=True))
                self.LRP_W_d.append(self.ny_projector_d.projection_matrix)
                self.LRP_b_d.append(None)

        # PROIEZIONE NEI LAYERS SUCCESSIVI DELLA KDA:
        for i, hl_conf in enumerate(self.hl_list):         # hl_conf sono oggetti HiddenLayerConf
            hl_size = hl_conf.get_size()
            hl_size = hl_conf.get_size()
            hl_activation = hl_conf.get_activation()

            if self.type == "n" or self.type == "s" or self.type == "sd":
                W_s = tf.Variable(tf.truncated_normal(shape=[in_size, hl_size], stddev=0.1), name="hls-W-" + str(i))
                b_s = tf.Variable(tf.constant(0.1, shape=[hl_size]), name="hls-b-" + str(i))
                # Salvo i parametri per fare la LRP
                # self.LRP_input_tensors_s.append(tf.identity(self.hl_projection_s))
                self.LRP_W_s.append(W_s)
                self.LRP_b_s.append(b_s)
                # Aggiorno il vettore proiezione che diventa l'input del layer successivo
                self.hl_projection_s = tf.nn.dropout(self.hl_projection_s, self.dkp, seed=self.rnd_seed)
                ########################################################################################
                self.LRP_input_tensors_s.append(tf.identity(self.hl_projection_s))
                ########################################################################################
                self.hl_projection_s = activate(
                    tf.nn.xw_plus_b(self.hl_projection_s, W_s, b_s, name="hl-projection-" + str(i)), hl_activation)   # Aggiornamento del vettore proiettato (Attivazione dei neuroni)
                self.l2_loss += tf.nn.l2_loss(W_s)                                                                    # Ad ogni passo accumulo loss

            if self.type == "d" or self.type == "sd":
                W_d = tf.Variable(tf.truncated_normal(shape=[in_size, hl_size], stddev=0.1), name="hld-W-" + str(i))
                b_d = tf.Variable(tf.constant(0.1, shape=[hl_size]), name="hld-b-" + str(i))
                # Salvo i parametri per fare la LRP
                self.LRP_input_tensors_d.append(tf.identity(self.hl_projection_d))
                self.LRP_W_d.append(W_d)
                self.LRP_b_d.append(b_d)
                # Aggiorno il vettore proiezione che diventa l'input del layer successivo
                self.hl_projection_d = tf.nn.dropout(self.hl_projection_d, self.dkp, seed=self.rnd_seed)
                self.hl_projection_d = activate(
                    tf.nn.xw_plus_b(self.hl_projection_d, W_d, b_d, name="hl-projection-" + str(i)), hl_activation)
                self.l2_loss += tf.nn.l2_loss(W_d)

            in_size = hl_size  # Aggiorno le dimensioni dell'input dell'hidden layer successivo

        # CALCOLO DEL VETTORE DA MANDARE NEL LAYER DI OUTPUT (Add dropout):
        with tf.name_scope("dropout"):
            if self.type == "n" or self.type == "s" or self.type == "sd":
                self.dropped_out_s = tf.nn.dropout(self.hl_projection_s, self.dkp, seed=self.rnd_seed)

            if self.type == "d" or self.type == "sd":
                self.dropped_out_d = tf.nn.dropout(self.hl_projection_d, self.dkp, seed=self.rnd_seed)

        # CALCOLO DEL VETTORE DI OUTPUT CONTENENTE LO SCORE DI CIASCUNA CLASSE:
        with tf.name_scope("output"):
            proj_s = self.ny_projector_s.projection_matrix
            proj_d = self.ny_projector_d.projection_matrix

            if self.type == "n" or self.type == "s" or self.type == "sd":
                W_cl_s = tf.Variable(tf.truncated_normal([in_size, num_classes], stddev=0.1), name="Ws-classifier")
                b_cl_s = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="bs-classifier")
                ########################################################################################################
                # self.LRP_input_tensors_s.append(tf.identity(self.hl_projection_s))
                self.LRP_input_tensors_s.append(tf.identity(self.dropped_out_s))
                ########################################################################################################
                self.LRP_W_s.append(W_cl_s)
                self.LRP_b_s.append(b_cl_s)
                self.scores_s = tf.nn.xw_plus_b(self.dropped_out_s, W_cl_s, b_cl_s, name="scores_s")
                self.l2_loss += tf.nn.l2_loss(W_cl_s)

            if self.type == "d" or self.type == "sd":
                W_cl_d = tf.Variable(tf.truncated_normal([in_size, num_classes], stddev=0.1), name="Wd-classifier")
                b_cl_d = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="bd-classifier")
                self.LRP_input_tensors_d.append(tf.identity(self.dropped_out_d))
                self.LRP_W_d.append(W_cl_d)
                self.LRP_b_d.append(b_cl_d)
                if self.l2_lambda > 0:
                    self.l2_loss += tf.nn.l2_loss(tf.subtract(proj_s, proj_d))
                    self.l2_loss += tf.nn.l2_loss(W_cl_d)
                self.scores_d = tf.nn.xw_plus_b(self.dropped_out_d, W_cl_d, b_cl_d, name="scores_d")

            # IN BASE AL TIPO DI RETE, FISSO GLI SCORES FINALI:
            if self.type == "n" or self.type == "s":
                self.scores = self.scores_s
            if self.type == "d":
                self.scores = self.scores_d
            if self.type == "sd":
                self.scores = tf.multiply(self.scores_s, self.scores_d)

            soft_max = tf.nn.softmax(self.scores)
            self.pred_argmax = tf.argmax(soft_max, 1)  # Softmax ==> Scores normalizzati
            self.gold_argmax = tf.argmax(self.input_y, 1)                       # tf.argmax returns a tensor of output_type type
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            self.softmax = soft_max

        with tf.name_scope("LRP"):
            # print(tf.contrib.training.train)
            if self.is_test is not None:
                if self.type == "n" or self.type == "s":
                    # R_s = self.scores
                    R_s = simple_lrp_softmax(self.scores, soft_max)
                    LRP_size = len(self.LRP_input_tensors_s)
                    for i in range(LRP_size - 1, -1, -1):
                        input_tensor = self.LRP_input_tensors_s[i]
                        weights = self.LRP_W_s[i]
                        biases = self.LRP_b_s[i]
                        R_s = simple_lrp_linear(R_s, input_tensor, weights, biases=biases)
                        # R = epsilon_lrp(R, 1, input_tensor, weights, biases=biases)
                    self.R = R_s



        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = softmax_cross_entropy(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + self.l2_lambda * self.l2_loss

        with tf.name_scope("f1"):
            self.corrects = {}
            self.preds = {}
            self.golds = {}
            self.precisions = {}
            self.recalls = {}
            self.f1s = {}
            for label in range(num_classes):
                self.preds[label] = tf.reduce_sum(tf.cast(tf.equal(self.pred_argmax, label), tf.int32))          # tf.equal is element-wise, it returns a boolean tensor
                self.golds[label] = tf.reduce_sum(tf.cast(tf.equal(self.gold_argmax, label), tf.int32))          # tf.cast casts a tensor to a new type ( i.e. da booleani ad interi)

                self.corrects[label] = tf.reduce_sum(
                    tf.cast(tf.logical_and(tf.equal(self.pred_argmax, self.gold_argmax),
                                           tf.equal(self.gold_argmax, label)), tf.int32))

                self.precisions[label] = tf.truediv(self.corrects[label], self.preds[label])
                self.recalls[label] = tf.truediv(self.corrects[label], self.golds[label])
                self.f1s[label] = tf.truediv(2 * tf.multiply(self.precisions[label], self.recalls[label]),
                                             tf.add(self.precisions[label], self.recalls[label]))

            self.f1 = tf.reduce_mean(list(self.f1s.values()))

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
