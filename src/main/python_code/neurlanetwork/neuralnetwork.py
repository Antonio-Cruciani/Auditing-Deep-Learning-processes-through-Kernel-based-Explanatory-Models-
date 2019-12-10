
from src.main.python_code.layers.hidden_layer_configuration import HiddenLayerConf
from src.main.python_code.layers.proj_matrix import ProjectionMatrix
from src.main.python_code.layers.ny_projector import NyProjector
from src.main.python_code.layers.kernel_based_deep_architecture import KernelBasedDeepArchitecture
from src.main.python_code.labelsofclasses.labels import labels_maps
from src.main.python_code.landmarks.landmarks import Landmarks
from src.main.python_code.fileoperations.read_files import get_landmarks_from_file,get_labels_from_file
from src.main.python_code.data.instance import Instance
from src.main.python_code.learning.train_LRP_MLP_QC import train_LRP_mlp_qc
from src.main.python_code.neurlanetwork.result import Result


import sys
import numpy as np
import tensorflow as tf



landmarks_directory = "src/main/resources/"

labels_directory = "src/main/resources/"

model_directory = "models/"

k_parameter = 10

landmarks = Landmarks(get_landmarks_from_file(landmarks_directory))
label_map,label_inverted_map = get_labels_from_file(labels_directory)
labels = labels_maps(label_map,label_inverted_map)

# Funzione che carica le flags
def get_flags():
    # Data parameters
    def del_all_flags(FLAGS):
        flags_dict = FLAGS._flags()
        key_list = [keys for keys in flags_dict]
        for keys in key_list:
            FLAGS.__delattr__(keys)

    del_all_flags(tf.flags.FLAGS)

    tf.flags.DEFINE_string("proj_matrix_path",
                           landmarks_directory + "proj_matrix.txt.gz",
                           "The path of the projection matrix")

    # Model parameters
    tf.flags.DEFINE_float("dkp", 0.8, "Dropout keep probability")
    tf.flags.DEFINE_float("l2_reg", 0.0, "L2 regularizer lambda")

    tf.flags.DEFINE_string("hl_conf",
                           str(landmarks.get_size_of_landmarks()) + ":relu",
                           "The hidden layers conf in the format hl1:act1,hl2:act2....")

    # General parameters
    tf.flags.DEFINE_integer("rnd_seed", 10, "The seed for the random stuff")
    tf.flags.DEFINE_integer("processors_num", 4, "Number of processors")

    # Training parameters
    tf.flags.DEFINE_integer("batch_size", 256, "Batch Size (default: 50)")
    tf.flags.DEFINE_integer("num_epochs", 500, "Number of training epochs (default: 50)")
    tf.flags.DEFINE_float("learning_rate", 0.003, "set the learning rate for the AdamOptimizer (default: 10e-4)")

    # Other Parameters
    tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

    tf.flags.DEFINE_string("pred_path",
                           landmarks_directory + "output/tmp_LRP_pred.txt",
                           "The path of the predictions")
    tf.flags.DEFINE_string("R_path",
                           landmarks_directory + "output/tmp_R.txt",
                           "The path of the R")
    tf.flags.DEFINE_string("Softmax_path",
                           landmarks_directory + "output/tmp_LRP_softmax.txt",
                           "The path of the Softmax scores")
    tf.flags.DEFINE_string("port", "port", "port")
    tf.flags.DEFINE_string("host", "host", "host")
    FLAGS = tf.flags.FLAGS
    FLAGS(sys.argv)
    return (FLAGS)

def training_new_model():
    train_LRP_mlp_qc(landmarks_directory, landmarks.get_size_of_landmarks())


# Funzione che carica la rete neurale precentemente salvata

def create_model():

    FLAGS = get_flags()


    np.set_printoptions(threshold=np.inf)


    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")



    with tf.Graph().as_default():
        for attr, value in sorted(FLAGS.__flags.items()):
            print("{}={}".format(attr.upper(), value))
        print("")
        np.random.seed(FLAGS.rnd_seed)
        tf.set_random_seed(FLAGS.rnd_seed)

        # Load data
        print("Loading Training Data...")


        # build hidden layers configurations
        hl_list = []
        if FLAGS.hl_conf:
            for hl_conf in FLAGS.hl_conf.split(","):
                sp = hl_conf.split(":")
                hl_list.append(HiddenLayerConf(int(sp[0]), sp[1]))


        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement, inter_op_parallelism_threads=FLAGS.processors_num,
            intra_op_parallelism_threads=FLAGS.processors_num)
        #sess = tf.Session(config=session_conf)
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            for attr, value in sorted(FLAGS.__flags.items()):
                print("{}={}".format(attr.upper(), value))
            print("")
            #print ("\n CIAOOO")
            proj_matrix_static = ProjectionMatrix(FLAGS.proj_matrix_path, trainable=False)
            ny_projector_s = NyProjector(projection_matrix=proj_matrix_static)

            proj_matrix_dynamic = ProjectionMatrix(FLAGS.proj_matrix_path, trainable=True)
            ny_projector_d = NyProjector(projection_matrix=proj_matrix_dynamic)

            ny_mlp = KernelBasedDeepArchitecture(nn_typeA="s", projector_s=ny_projector_s,
                                                 # ny_mlp.input_x e ny_mpl.input_y vengono
                                                 projector_d=ny_projector_d, hl_conf_list=hl_list,
                                                 l2_lambda=FLAGS.l2_reg,  # assegnati nelle feed_dict di train_step(x,y)
                                                 num_classes=len(label_map),
                                                 rnd_seed=FLAGS.rnd_seed,landmarks_size=landmarks.get_size_of_landmarks())  # e di test_step(x,y)
        print("LOADING Neural Network \n")
        saver = tf.train.Saver()
        # Restore variables from disk.
        saver.restore(sess, model_directory+"saved_model")
        print("Model restored.")
        return(ny_mlp,sess)


# Metodo che classifica un vettore in input
def classify(mlp,nn_session,c_vector,question):

    tt_x, tt_y = get_arrays_from(c_vector)

    feed_dict = {
        mlp.input_x: tt_x[0],
        mlp.input_y: tt_y,
        mlp.dkp: 1.0,
        mlp.is_test: 1.0
    }
    tt_loss, tt_acc, tt_preds, tt_R, tt_scores, tt_softmax =nn_session.run([ mlp.loss,mlp.accuracy, mlp.predictions, mlp.R, mlp.scores,mlp.softmax], feed_dict)

    label_inverted_map = labels.get_inverted_label_map()
    
    predicted = label_inverted_map[tt_preds[0]]
    #trasforma tt_R in lista
    tt_R_list = tt_R.tolist()

    prediction = Result()
    prediction.set_k_parameter(k_parameter)
    prediction.set_information(question)
    prediction.set_predicted_class(predicted)
    indici_ordinati = sorted(range(len(tt_R_list[0])), key=lambda k: tt_R_list[0][k], reverse=True)
    k_most_relevant_indexes = indici_ordinati[0:k_parameter]
    list_of_landmarks = landmarks.get_landmarks()
    c = len(list_of_landmarks) - k_parameter
    k_less_relevant_indexes = indici_ordinati[c:len(list_of_landmarks)]
    positive_consistent = []
    negative_consistent = []
    k_most_relevant_landmarks = []
    k_less_relevant_landmarks = []
    # Prendi la lista dei k landmark positivi e negativi
    for i in k_most_relevant_indexes:
        k_most_relevant_landmarks.append(list_of_landmarks[i])

    for i in k_less_relevant_indexes:
        k_less_relevant_landmarks.append(list_of_landmarks[i])
    for i in k_most_relevant_landmarks:
        if (i[0] == predicted):
            positive_consistent.append(i)

    for i in reversed(k_less_relevant_landmarks):
        if (i[0] != predicted):
            negative_consistent.append(i)

    prediction.set_k_positive_landmarks(positive_consistent)

    prediction.set_k_complement_landmarks(negative_consistent, labels.get_label_map())

    return prediction


def get_arrays_from(c_vector):
    label_map = labels.get_label_map()

    x = [p.get_vector() for p in c_vector]  # lista di numpy.ndarray costituite da un solo int32
    y = [p.get_vector_label(label_map) for p in
         c_vector]  # lista di array di float64 lunghe quanto label_map e costituite da un 1.0 e tutti 0.0
    return np.array(x),np.array(y) # Restituisce una numpy.ndarray di numpy.ndarray



def get_ny_proj(mlp,nn_session,projected_vector,question):

    label = "ABBR"
    c_vector= [Instance(projected_vector,label)]
    prediction =classify(mlp,nn_session,c_vector,question)

    return prediction

# Funzione main nella quale viene addestrata la rete neurale
def main():
    training_new_model()


if __name__ == '__main__':
    main()


