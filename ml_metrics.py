
class MLMetrics:
    def __init__(self):
        self.true_pos = 0
        self.false_neg = 0
        self.false_pos = 0
        self.true_neg = 0

    def update(self, predicted, actual):
        '''
        Determine the true/false positives/negatives
        :param bool predicted: The value predicted by the ML system
        :param bool actual: The actual value from the evaluation data
        :return:
        '''
        if predicted:
            if actual:
                self.true_pos += 1
            else:
                self.false_pos += 1
        else:
            if actual:
                self.false_neg += 1
            else:
                self.true_neg += 1

    def calc_precision(self):
        # Percent of positive predictions which were correct predictions
        if(self.true_pos + self.false_pos) > 0:
            return self.true_pos/(self.true_pos + self.false_pos)
        else:
            return -1

    def calc_recall(self):
        # Percent of overall positives predicted correctly
        if(self.true_pos + self.false_neg) > 0:
            return self.true_pos/(self.true_pos + self.false_neg)
        else:
            return -1

    def calc_f1_score(self):
        prec = self.calc_precision()
        rec = self.calc_recall()
        if(prec + rec) > 0:
            return 2 * prec * rec / (prec + rec)
        else:
            return -1

    def calc_accuracy(self):
        # Percent of actions predicted correctly
        if(self.true_pos + self.true_neg + self.false_pos + self.false_neg) > 0:
            return (self.true_pos + self.true_neg) / (self.true_pos + self.true_neg + self.false_pos + self.false_neg)

    def calc_click_through_rate(self):
        # Percent of notifications users clicked
        if (self.true_pos + self.true_neg + self.false_pos + self.false_neg) > 0:
            return (self.true_pos + self.false_neg) / (self.true_pos + self.true_neg + self.false_pos + self.false_neg)

