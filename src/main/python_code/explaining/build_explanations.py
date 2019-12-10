from src.main.python_code.explaining.Explanation import explanation_model
from flask import Markup
from random import choice

class explenation_qc(explanation_model):

    def build_explanation_positive_singleton(self,prediction):


        original_question = prediction.get_information()
        positive = prediction.get_k_positive_landmarks()
        predicted_class = prediction.get_predicted_class()
        positive_singleton = None

        if not positive:
            positive_singleton = Markup("I think " +"<i>" +original_question +"</i>" +" refers to a " +" <u>" +"<font face=\"courier\" color=\"black\">"+ predicted_class +"</font>"+"< /u >" +" but I don't know why")
        else:
            positive_singleton = Markup("I think " + "<i>" + original_question + "</i>" + " refers to a " + " <u>" + "<font face=\"courier\" color=\"black\">" + predicted_class + "</font>" + "</u>" + " since it recalls me of " + "<strong><i>" + positive[0][1]+ "</i></strong>" + " that also refers to a " + "<u>" + "<font face=\"courier\" color=\"black\">" +positive[0][0] + "</font>" + "</u>" + ".")


        return positive_singleton

    def build_explanation_negative_singleton(self, prediction):
        original_question = prediction.get_information()
        predicted_class = prediction.get_predicted_class()
        negative = None
        negative_singleton = None
        complement_landmarks = prediction.get_k_complement_landmarks()
        not_null_list = []
        for i in complement_landmarks.keys():
            if (complement_landmarks[i] != []):
                for j in complement_landmarks[i]:
                    not_null_list.append([i, j])
        if not_null_list != []:
            negative = choice(not_null_list)

        if negative is None:
            negative_singleton = Markup("I think " +"<i>" +original_question +"</i>" + " does not refers to a  "+"<u>"+"<font face=\"courier\" color=\"black\">"+ predicted_class +"</font>"+"</u>"+" but I don't know why.")
        else:
            negative_singleton = Markup("I think " +"<i>"+ original_question +"</i>" +" does not refers to a " +"<u>"+"<font face=\"courier\" color=\"black\">"+  negative[0] +" </font>"+"</u>" +" since it does not recall me of " +"<strong> <i>"+ \
                                     negative[1] +"</i></strong>"+ " that refers to a " +"<u>" +"<font face=\"courier\" color=\"black\">"+ negative[0]+ "</font>"+"</u>" + ".")

        return negative_singleton

    def build_explanation_positive_conjunctive(self, prediction,positive_number):

        sample = None
        #index_sample = None
        original_question = prediction.get_information()
        positive_landmarks = prediction.get_k_positive_landmarks()
        predicted_class = prediction.get_predicted_class()
        positive = []

        positive_conjunctive = None

        if positive_landmarks and (len(positive_landmarks) - positive_number) == 0:
            positive = positive_landmarks
        elif positive_landmarks and (len(positive_landmarks) - positive_number) < 0:
            positive = positive_landmarks
        elif positive_landmarks and (len(positive_landmarks) - positive_number) > 0:
            positive = positive_landmarks[:positive_number]



        if(not positive):
            positive_conjunctive = Markup("I think " +"<i>"+ original_question +"</i>" +" refers to a " +"<u>" +"<font face=\"courier\" color=\"black\">"+ predicted_class +"</font>"+"</u>" +" but I don't know why")
        else:
            positive_conjunctive_head = "I think " + "<i>" + original_question + "</i>" + " refers to a " + "</u>" + "<font face=\"courier\" color=\"black\">" + predicted_class + "</font>" + "</u>" + " since it recalls me of  "
            positive_conjunctive_tail = ""
            j = 0
            for i in positive:
                positive_conjunctive_tail += "<strong><i>" + i[
                    1] + "</strong></i>" + " that also refers to a " + "<u>" + "<font face=\"courier\" color=\"black\">" + \
                                            i[0] + "</font>" + "</u>"
                if (j < len(positive) - 1):
                    positive_conjunctive_tail += " and it also recalls me of "
                j+=1
            positive_conjunctive = Markup(positive_conjunctive_head + positive_conjunctive_tail)



        return positive_conjunctive

    def build_explanation_negative_conjunctive(self,prediction, negative_number):
        sample = None
        index_sample = None
        original_question = prediction.get_information()
        predicted_class = prediction.get_predicted_class()
        negative = []
        negative_conjunctive = None

        complement_landmarks = prediction.get_k_complement_landmarks()

        not_null_list = []
        for i in complement_landmarks.keys():
            if (complement_landmarks[i] != []):
                for j in complement_landmarks[i]:
                    not_null_list.append([i, j])
        i = 0
        while (len(not_null_list) > 0 and i < negative_number):
            sample = choice(not_null_list)
            negative.append(sample)
            not_null_list.remove(sample)
            i += 1

        if (not negative):
            negative_conjunctive = Markup(
                "I think " + "<i>" + original_question + "</i>" + " does not refers to a  " + "<u>" + "<font face=\"courier\" color=\"black\">" + predicted_class + "</font>" + "</u>" + " but I don't know why.")
        else:
            negative_conjunctive_head = "I think " + "<i>" + original_question + "</i>" + " does not refers to a "
            negative_conjunctive_tail = ""
            j = 0
            for i in negative:
                negative_conjunctive_tail += "<u>" + "<font face=\"courier\" color=\"black\">" + i[
                    0] + "</font>" + "</u>" + " since it does not recall me of " + "<strong><i>" + i[
                                                 1] + "</strong></i>" + " that refers to a " + "<u>" + "<font face=\"courier\" color=\"black\">" + \
                                             i[0] + "</font>" + "</u>"
                if (j < len(negative) - 1):
                    negative_conjunctive_tail += " and does not refers to a "
                j += 1
            negative_conjunctive = Markup(negative_conjunctive_head + negative_conjunctive_tail)

        return negative_conjunctive



    # Metodo che costruisce l'explanation contrastive
    def build_explanation_positive_contrastive(self, prediction):
        sample = None
        index_sample = None
        original_question = prediction.get_information()
        positive_landmarks = prediction.get_k_positive_landmarks()

        predicted_class = prediction.get_predicted_class()
        positive = positive_landmarks

        negative = []
        negative_list = []
        positive_contrastive = None
        negative_contrastive = None
        complement_landmarks = prediction.get_k_complement_landmarks()
        not_null_list = []
        for i in complement_landmarks.keys():
            if (complement_landmarks[i] != []):
                for j in complement_landmarks[i]:
                    not_null_list.append([i, j])
        #i = 0

        # while (len(not_null_list) > 0 and i < prediction.get_k_parameter()):
        #     sample = choice(not_null_list)
        #     negative_list.append(sample)
        #     not_null_list.remove(sample)
        #     i += 1


        negative = choice(not_null_list)



        if (positive == []):
            positive_contrastive = Markup("I think " +"<i>"+ original_question +"</i>" +" refers to a " +"<u>" +"<font face=\"courier\" color=\"black\">"+ predicted_class +"</font>"+"</u>" +" but I don't know why")
        else:
            positive_contrastive = Markup("I think "+"<i>" + original_question +"</i>"+ " refers to a " +"<u>"+ "<font face=\"courier\" color=\"black\">" +predicted_class +"</font>" +"</u>"+ " since it recalls me of "+"<strong><i>" + \
                                          positive[0][1] +"</i></strong>"+ " that also refers to a " +"<u>"+"<font face=\"courier\" color=\"black\">" + positive[0][0]
                                       +"</font>"+"</u>" + " and because it does not recall me of " +"<strong><i>"+  negative[1] +"</i></strong>"+ " which refers to a " +"<u>"+"<font face=\"courier\" color=\"black\">" +  negative[0] +"</font>"+"</u>"+ " instead.")
        return positive_contrastive

    def build_explanation_negative_contrastive(self, prediction):

        sample = None
        index_sample = None
        original_question = prediction.get_information()
        positive_landmarks = prediction.get_k_positive_landmarks()
        positive = positive_landmarks

        predicted_class = prediction.get_predicted_class()
        negative = []
        negative_list = []
        positive_contrastive = None
        negative_contrastive = None
        complement_landmarks = prediction.get_k_complement_landmarks()
        not_null_list = []
        for i in complement_landmarks.keys():
            if (complement_landmarks[i] != []):
                for j in complement_landmarks[i]:
                    not_null_list.append([i, j])
        #i = 0
        # while (len(not_null_list) > 0 and i < prediction.get_k_parameter()):
        #     sample = choice(not_null_list)
        #     negative_list.append(sample)
        #     not_null_list.remove(sample)
        #     i += 1

        negative = choice(not_null_list)

        if (negative == []):
            negative_contrastive = Markup(
                "I think " + "<i>" + original_question + "</i>" + " does not refers to a  " + "<u>" + "<font face=\"courier\" color=\"black\">" + predicted_class + "</font>" + "</u>" + " but I don't know why.")
        else:
            negative_contrastive = Markup(
                "I think " + "<i>" + original_question + "</i>" + " does not refers to a " + "<u>" + "<font face=\"courier\" color=\"black\">" +
                negative[
                    0] + "</font>" + "</u>" + " since it does not recall me of " + "<strong><i>" + \
                negative[1] + "</strong></i>" + " that refers to a " + "<u>" + "<font face=\"courier\" color=\"black\">" +
                negative[
                    0] + "</font>" + "</u>" + ", but I think it is " + "<u>" + "<font face=\"courier\" color=\"black\">" + predicted_class + "</font>" + "</u>" + " because it recalls me of " + "<strong><i>" + \
                positive[0][
                    1] + "</strong></i>" + " which also refers to a " + "<u>" + "<font face=\"courier\" color=\"black\">" +
                positive[0][0] + "</font>" + "</u>" + ".")

        return negative_contrastive