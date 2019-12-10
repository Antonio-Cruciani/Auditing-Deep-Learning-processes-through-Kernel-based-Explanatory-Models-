class labels_maps:

    def __init__ (self, labels,inverted_labels):
        self.label_inverted_map = labels
        self.label_map = inverted_labels


    def get_inverted_label_map(self):
        return self.label_inverted_map

    def get_label_map(self):
        return self.label_map

    def get_labels(self):
        return(self.label_map.keys())
