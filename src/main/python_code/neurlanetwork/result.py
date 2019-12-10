

class Result:

    def __init__(self):
        self.information = None
        self.predicted_class = None
        self.k_positive_landmarks = []
        self.k_complement_landmaks = {}
        self.k_parameter = None

    def get_k_parameter(self):
        return self.k_parameter

    def get_information(self):
        return self.information

    def get_predicted_class(self):
        return self.predicted_class

    def get_k_positive_landmarks(self):
        return self.k_positive_landmarks

    def get_k_complement_landmarks(self):
        return self.k_complement_landmaks

    def set_k_parameter(self, k):
        self.k_parameter = k

    def set_information(self,question):
        self.information = question

    def set_predicted_class(self,predicted_label):
        self.predicted_class = predicted_label

    def set_k_positive_landmarks(self,list_of_positive_landmarks):
        self.k_positive_landmarks = list_of_positive_landmarks

    def set_k_complement_landmarks(self, list_of_negative_landmarks, labels):
        for i in labels:
            self.k_complement_landmaks[i] = []

        for i in list_of_negative_landmarks:
            self.k_complement_landmaks[i[0]].append(i[1])

