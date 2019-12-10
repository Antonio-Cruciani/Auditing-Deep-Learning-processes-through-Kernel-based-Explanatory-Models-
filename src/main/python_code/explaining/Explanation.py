# import numpy as np
#
#
# MODEL SINGLETON QC
#
# Positive:"I think @#@QUESTION_PLACEHOLDER@#@ refers to a @#@PREDICTED_CLASS_PLACEHOLDER@#@ since it recalls me of @#@POSITIVE_LANDMARK_PLACEHOLDER@#@ that also refers to a @#@PREDICTED_CLASS_PLACEHOLDER@#@."
#
# Negative:"I think @#@QUESTION_PLACEHOLDER@#@ does not refers to a @#@REJECTED_CLASS_PLACEHOLDER@#@ since it does not recall me of @#@NEGATIVE_LANDMARK_PLACEHOLDER@#@ that refers to a @#@REJECTED_CLASS_PLACEHOLDER@#@."
#
#
# MODEL CONTRASTIVE QC
#
# Positive:"I think @#@QUESTION_PLACEHOLDER@#@ refers to a @#@PREDICTED_CLASS_PLACEHOLDER@#@ since it recalls me of @#@POSITIVE_LANDMARK_PLACEHOLDER@#@ that also refers to a @#@PREDICTED_CLASS_PLACEHOLDER@#@ and because it does not recall me of @#@NEGATIVE_LANDMARK_PLACEHOLDER@#@ which refers to a @#@REJECTED_CLASS_PLACEHOLDER@#@ instead."
#
# Negative:""I think @#@QUESTION_PLACEHOLDER@#@ does not refers to a @#@REJECTED_CLASS_PLACEHOLDER@#@ since it does not recall me of @#@NEGATIVE_LANDMARK_PLACEHOLDER@#@ that refers to a @#@REJECTED_CLASS_PLACEHOLDER@#@, but I think it is @#@PREDICTED_CLASS_PLACEHOLDER@#@ because it recalls me of @#@POSITIVE_LANDMARK_PLACEHOLDER@#@ which also refers to a @#@PREDICTED_CLASS_PLACEHOLDER@#@."


from abc import ABC, abstractmethod

class explanation_model(ABC):

    def build_explanation_positive_singleton(self):
        raise NotImplementedError("Error you have to implement this class")

    def build_explanation_negative_singleton(self):
        raise NotImplementedError("Error you have to implement this class")

    def build_explanation_positive_conjunctive(self):
        raise NotImplementedError("Error you have to implement this class")

    def build_explanation_negative_conjunctive(self):
        raise NotImplementedError("Error you have to implement this class")

    def build_explanation_positive_contrastive(self):
        raise NotImplementedError("Error you have to implement this class")

    def build_explanation_negative_contrastive(self):
        raise NotImplementedError("Error you have to implement this class")

    #def unpacks(self):
    #    raise NotImplementedError("Error you have to implement this class")

    #def validate_json(self):
    #    raise NotImplementedError("Error you have to implement this class")






