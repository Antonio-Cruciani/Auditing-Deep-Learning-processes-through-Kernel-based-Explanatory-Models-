import tensorflow as tf
from tensorflow import keras
import sys
import numpy as np
import datetime
import gzip
import math as mt
import xlsxwriter
from src.main.python_code.utils.classificationUtils import prep_sprdsheet
from src.main.python_code.layers.hidden_layer_configuration import HiddenLayerConf
from src.main.python_code.layers.proj_matrix import ProjectionMatrix
from src.main.python_code.layers.ny_projector import NyProjector
from src.main.python_code.layers.kernel_based_deep_architecture import KernelBasedDeepArchitecture
from src.main.python_code.layers.instance_embeddings import InstanceEmbedding
from src.main.python_code.data.instance import Instance
import json

def train_LRP_mlp_qc(file_dir, landmark_size):
    np.set_printoptions(threshold=np.inf)

    # Main parameters (they have default values, but will be replace by the command FLAGS._parse_flags()
    tf.flags.DEFINE_string("nn_type", "s",
                           'Choose the configuration of the projectors: n is no_nystrom, s is static, '
                           'd is dynamic, sd is static and dynamic')
    # Data parameters
    dir_name_header = file_dir
    tf.flags.DEFINE_string("c_matrix_path", dir_name_header + "c_matrix.txt.gz",
                           "The path of the c matrix")

    tf.flags.DEFINE_string("proj_matrix_path",
                           dir_name_header + "proj_matrix.txt.gz",
                           "The path of the projection matrix")

    tf.flags.DEFINE_string("train_path",
                            dir_name_header + "train.txt.gz",
                           "The path of the dataset")

    # tf.flags.DEFINE_string("train_path",
    #                       dir_name_header + "train_ids_class.txt.gz",
    #                       "The path of the dataset")





    tf.flags.DEFINE_string("test_path",
                           dir_name_header + "test.txt.gz",
                           "The path of the dataset")

    # Model parameters
    tf.flags.DEFINE_float("dkp", 0.8, "Dropout keep probability")
    tf.flags.DEFINE_float("l2_reg", 0.0, "L2 regularizer lambda")


    tf.flags.DEFINE_string("hl_conf",
                           str(landmark_size)+":relu",
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
                           dir_name_header+"output/tmp_LRP_pred.txt",
                           "The path of the predictions")
    tf.flags.DEFINE_string("R_path",
                           dir_name_header + "output/tmp_R.txt",
                           "The path of the R")
    tf.flags.DEFINE_string("Softmax_path",
                           dir_name_header + "output/tmp_LRP_softmax.txt",
                           "The path of the Softmax scores")

    FLAGS = tf.flags.FLAGS
    FLAGS(sys.argv)
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")

    label_map = {"DESC": 0, "ENTY": 1, "ABBR": 2, "HUM": 3, "LOC": 4, "NUM": 5}
    label_inverted_map = {0: "DESC", 1: "ENTY", 2: "ABBR", 3: "HUM", 4: "LOC", 5: "NUM"}


    def load_vect(path,train_path):

        #c_matrix_list = []
        data = []
        #labels =[]
        app = []

        test = []

        test_array = []
        train = []
        train_array = []
        with gzip.open(train_path,'rt') as g:
            train_file = json.load(g)
            i = 0
            for line in train_file:
                app.append(str(i)+"\t"+line["label"]+"\t"+line["question"])
                riga = line["vector"].split(" ")
                test.append(riga)
                instance_vector = np.asarray(riga,dtype=np.float32)
                #print(riga)
                test_array.append(Instance(instance_vector,line["label"]))
                i+=1
        i = 0
        with gzip.open(path,'rt') as f:
            train_file = json.load(f)
            for line in train_file:
                app.append(str(i) + "\t" + line["label"] + "\t" + line["question"])
                riga = line["vector"].split(" ")
                train.append(riga)
                instance_vector = np.asarray(riga, dtype=np.float32)
                train_array.append(Instance(instance_vector,line["label"]))
                i += 1


        return train_array,test_array


        #offset = len(app)




        # i=0
        # with gzip.open(path,'rt') as f:
        #     for line in f:
        #         if (i==0):
        #             # Getting the dimensionality of the matrix\vector (n,m) from the frist line of the c_matrix file
        #             split = line.split(' ')
        #             #print("AHOOOO\n")
        #             #print(split)
        #
        #             n = int(split[0])
        #             m = int(split[1])
        #             i+=1
        #             #print (n,m)
        #         else:
        #             # Getting the vectors
        #             split = line.split('\t')
        #             # Getting the labels
        #             #print("CIAOOO\n")
        #             #print(split)
        #             string_label = split[1]
        #             #labels.append(split[1])
        #             split_numbers = split[3].split(',')
        #             # Deleting the \n character from the list
        #             app = split_numbers[m-1].split('\n')
        #             split_numbers[m-1]=app[0]
        #             # Converting the list of numbers in a np array of floats
        #             instance_vector = np.asarray(split_numbers,dtype=np.float32)
        #             # Creating a list of list of numbers that represent the c_matrix
        #             #c_matrix_list.append(split_numbers)
        #             # Appending an Istance obj to a list
        #             data.append(Instance(instance_vector,string_label))
        #             #print (split_numbers)
        #             #exit()
            # Creating a numpy matrix from the list 
            #c_matrix_npa = np.asarray(c_matrix_list, dtype=np.float32)
           # return data[0:offset],data[offset+1:len(data)]
            #print (labels)
            #print ("\n")
            #print (c_matrix_npa)
            #exit()
            #for line in f:
            #    if (i>0):
            #        content = line.readlines()
  

    # FUNZIONE PER CARICARE I DATI: restituiti come una lista di oggetti Istance
    #    def load_data(path):
    #        data = []
    #        # with gzip.open(path, 'rb') as f:
    #        with gzip.open(path, 'rt') as f:
    #            for line in f:
    #                split = line.split("\t")
    #                instance_id = int(split[0])
    #
    #                string_label = str(split[1].strip())
    #                data.append(Instance(instance_id, string_label))

    #        return data  # data e' una lista di Instance

    # FUNZIONE DI SPLITTING DEL DATASET:
    def split_data(dataset, split_f=0.8):  # dataset e' una lista di Instances
        np.random.shuffle(dataset)  # np.random.shuffle() cambia direttamente dataset
        split_size = int(len(dataset) * split_f)
        return dataset[0:split_size], dataset[split_size:]

    # FUNZIONE CHE CONVERTE IL DATASET DA UNA LISTA DI ISTANZE AD UNA ARRAY:
    def get_arrays_from(dataset):

        x = [np.asarray(p.get_vector()) for p in dataset]  # lista di numpy.ndarray costituite da un solo int32
        y = [p.get_vector_label(label_map) for p in
             dataset]  # lista di array di float64 lunghe quanto label_map e costituite da un 1.0 e tutti 0.0
        
        return np.array(x), np.array(y)  # Restituisce una numpy.ndarray di numpy.ndarray

    def run_experiment(a_train, a_dev, a_test, a_ny_mlp,
                       a_sess):  # a_train,a_dev,a_test = list di Instances, a_ny_npl = KernelBasedDeepArchitecture, sess = tf.Session
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        grads_and_vars = optimizer.compute_gradients(
            a_ny_mlp.loss)  # It returns a list of (gradient, variable) pairs where "gradient" (can be a tensor or None) is the gradient for "variable"
        train_op = optimizer.apply_gradients(grads_and_vars,
                                             global_step=global_step)  # Apply gradients to variables. It returns an Operation that applies gradients.
        # If global_step was not None, that operation also increments global_step
        # Initialize all variables
        a_sess.run(tf.global_variables_initializer())  # Quali variabili globali?

        def train_step(x, y):  # x e y sono numpy.ndarray di numpy.ndarray
            """
            A training step
            """
            feed_dict = {
                ny_mlp.input_x: x,
                ny_mlp.input_y: y,
                ny_mlp.dkp: FLAGS.dkp  # per fare if della LRP
            }
            _, step, loss, acc = sess.run(
                [train_op, global_step, a_ny_mlp.loss, a_ny_mlp.accuracy],
                feed_dict)  # train_op e' la differenza con test_spet(x,y)
            return step, loss, acc

        def dev_step(x, y):
            """
            A test step
            """
            feed_dict = {
                ny_mlp.input_x: x,
                ny_mlp.input_y: y,
                ny_mlp.dkp: 1.0

            }
            step, loss, acc, preds = sess.run(
                [global_step, a_ny_mlp.loss, a_ny_mlp.accuracy, a_ny_mlp.predictions], feed_dict)
            return step, loss, acc, preds

        def test_step(x, y):
            """
            A test step
            """
            feed_dict = {
                #ny_mlp.input_x: x,
                ny_mlp.input_x: x,
                ny_mlp.input_y: y,
                ny_mlp.dkp: 1.0,
                ny_mlp.is_test: 1.0
            }
            step, loss, acc, preds, R, scores, softmax = sess.run(
                [global_step, a_ny_mlp.loss, a_ny_mlp.accuracy, a_ny_mlp.predictions, a_ny_mlp.R, a_ny_mlp.scores,
                 a_ny_mlp.softmax], feed_dict)
            return step, loss, acc, preds, R, scores, softmax

        # Getting Datasets as Numpy Arrays
        data_size = len(a_train)
        #print("\n A TRAIN ____\n")
        #print(a_train)
        # Dalle liste di Instances del TrainingSet, DevelopmentSet e TestSet, ottengo le array di array di ids e labels per ciascun data set
        tr_x, tr_y = get_arrays_from(a_train)
        dv_x, dv_y = get_arrays_from(a_dev)
        tt_x, tt_y = get_arrays_from(a_test)


        # Initializing the quantities useful for the training phase
        best_step = 0
        best_dev_loss = 0.0
        best_dev_acc = -1.0
        best_test_loss = 0.0
        best_test_acc = -1.0
        best_preds = None

        shuffle_indices = np.random.permutation(np.arange(data_size))

        for epoch in range(FLAGS.num_epochs):
            if data_size % FLAGS.batch_size == 0:
                num_batches_per_epoch = int(data_size / FLAGS.batch_size)
            else:
                num_batches_per_epoch = int(data_size / FLAGS.batch_size) + 1

            x_s = tr_x[shuffle_indices]  # Sono ancora array di array
            y_s = tr_y[shuffle_indices]

            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * FLAGS.batch_size
                end_index = min((batch_num + 1) * FLAGS.batch_size, data_size)

                cur_batch_x = x_s[start_index: end_index]
                cur_batch_y = y_s[start_index: end_index]
                #print (type(cur_batch_x))
                #exit()
                step, loss, acc = train_step(cur_batch_x, cur_batch_y)
                time_str = datetime.datetime.now().isoformat()
                if step % 10 == 0:
                    print("{} Training: epoch {} step {} loss {:g} accuracy {:g}".format(time_str, epoch, step, loss,
                                                                                         acc))

            dv_step, dv_loss, dv_acc, _ = dev_step(dv_x, dv_y)
            # dv_step, dv_loss, dv_acc, _, dv_R = test_step(dv_x, dv_y)

            if dv_acc > best_dev_acc:
                tt_step, tt_loss, tt_acc, tt_preds, tt_R, tt_scores, tt_softmax = test_step(tt_x, tt_y)
                best_step = dv_step
                best_dev_acc = dv_acc
                best_dev_loss = dv_loss
                best_test_acc = tt_acc
                best_test_loss = tt_loss
                best_preds = tt_preds
                # best_dev_R = dv_R
                best_test_R = tt_R
                best_softmax = tt_softmax
                best_scores = tt_scores
                #if (epoch >= FLAGS.num_epochs/2 ):
                if (epoch >=1 ):
                    saver = tf.train.Saver() 
                    saver.save(a_sess, 'models/saved_model')
                    #model.save("models/my_trained_model.ckpt")
                    #saver.save(a_sess,'models/my_model_prova.h5')
                    #saver.save(a_sess, 'models/model.ckpt')
            else:
                tt_step, tt_loss, tt_acc, tt_preds = dev_step(tt_x, tt_y)

            time_str = datetime.datetime.now().isoformat()
            print(
                "{} Development: epoch {} step {} loss {:g} accuracy {:g}".format(time_str, epoch, dv_step, dv_loss,
                                                                                  dv_acc))
            print(
                "{} Testing: epoch {} step {} loss {:g} accuracy {:g}".format(time_str, epoch, tt_step, tt_loss,
                                                                              tt_acc))
        return best_step, best_dev_loss, best_dev_acc, best_test_loss, best_test_acc, best_test_R, best_scores, best_softmax, best_preds
        # return best_step, best_dev_loss, best_dev_acc, best_dev_R, best_test_loss, best_test_acc, best_test_R, best_preds

    with tf.Graph().as_default():
        np.random.seed(FLAGS.rnd_seed)
        tf.set_random_seed(FLAGS.rnd_seed)

        # Load data
        print("Loading Training Data...")
        #all_train_dataset = load_data(FLAGS.train_path)  # All_train_dataset e' una lista di Instances
        train_app,test = load_vect(FLAGS.train_path,FLAGS.test_path)
        train, dev = split_data(train_app, 0.8)  # train e dev sono liste di Instances
        #train,test = split_data(train_app, 0.8) # calcolo il test set 
        #print("Loading Testing Data...")
        #test = load_vect(FLAGS.c_matrix_path)  # test e' una lista di Instances
        #test = load_data(FLAGS.test_path)  # test e' una lista di Instances
        #test_ids = [];  # list degli ids delle Instances di test
        #for instance in test:
        #    test_ids.append(instance.vector)

        #print("Training Data {trs}\tDevelopment Data {dvs}\tTesting Data {tss}"
        #      .format(trs=len(train), dvs=len(dev), tss=len(test)))

        # build hidden layers configurations
        hl_list = []
        if FLAGS.hl_conf:
            for hl_conf in FLAGS.hl_conf.split(","):
                sp = hl_conf.split(":")
                hl_list.append(HiddenLayerConf(int(sp[0]), sp[1]))

        # Training
        # ==================================================
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement, inter_op_parallelism_threads=FLAGS.processors_num,
            intra_op_parallelism_threads=FLAGS.processors_num)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            
            print ("\n CIAOOO\n")
            print(landmark_size)
            proj_matrix_static = ProjectionMatrix(FLAGS.proj_matrix_path, trainable=False)
            ny_projector_s = NyProjector(projection_matrix=proj_matrix_static)

            proj_matrix_dynamic = ProjectionMatrix(FLAGS.proj_matrix_path, trainable=True)
            ny_projector_d = NyProjector(projection_matrix=proj_matrix_dynamic)

            ny_mlp = KernelBasedDeepArchitecture(nn_typeA=FLAGS.nn_type, projector_s=ny_projector_s,
                                                 # ny_mlp.input_x e ny_mpl.input_y vengono
                                                 projector_d=ny_projector_d, hl_conf_list=hl_list,
                                                 l2_lambda=FLAGS.l2_reg,  # assegnati nelle feed_dict di train_step(x,y)
                                                 num_classes=len(label_map),
                                                 rnd_seed=FLAGS.rnd_seed,landmarks_size=landmark_size)  # e di test_step(x,y)

            # step, dv_loss, dv_acc, dv_R, tt_loss, tt_acc, tt_R, tt_preds = run_experiment(
            #     a_train=train, a_dev=dev, a_test=test, a_ny_mlp=ny_mlp, a_sess=sess)

            step, dv_loss, dv_acc, tt_loss, tt_acc, tt_R, tt_scores, tt_softmax, tt_preds = run_experiment(
                a_train=train, a_dev=dev, a_test=test, a_ny_mlp=ny_mlp, a_sess=sess)

            print("Best Dev: Step {:g} Loss {:g} Accuracy {:g}".format(step, dv_loss, dv_acc) +
                  " Best Test: Step {:g} Loss {:g} Accuracy {:g}".format(step, tt_loss, tt_acc))
            #Saving TF NN
                       # Salvare il modello, vedi la documentazione tf
            '''
            pred_file = open(FLAGS.pred_path, "w")
            for pred in tt_preds:
                pred_file.write(label_inverted_map[pred] + "\n")
            pred_file.close()

            
            workbook = xlsxwriter.Workbook(dir_name_header + 'output/LRP_Results.xlsx')

            dim = 500
            worksheet = workbook.add_worksheet(name= str(dim) + '_Landmarks')
            prep_sprdsheet(worksheet, dir_name_header)
            count_c = 2
            for R_row in tt_R:
                count_r = 1
                for score in R_row:
                    worksheet.write(count_r, count_c, score)
                    count_r += 1
                count_c +=1
            # count_c = 2
            # for pred in tt_preds:
            #     worksheet.write(201, count_c, label_inverted_map[pred])
            #     count_c += 1
            worksheet.autofilter(0,0,0,count_c)
            workbook.close()
            '''
