from sklearn.svm import SVC

from nti.data.algorithms.supervised import SupervisedModel

class SupportVectorMachine(SupervisedModel):
    """
    Abstraction of the SciKit Learn Support Vector Machine.
    """
    
    def __init__(self, data_frame, prediction_column, training_size = .7, **kwargs):
        super(SupportVectorMachine, self).__init__(data_frame, prediction_column, training_set_ratio=training_size)
        self.svc = SVC(**kwargs)
    
    def classify(self, inputs):
        pred_ans = self.svc.predict([inputs])
        return pred_ans
    
    def train(self):
        self.svc.fit(self._training_set_inputs, self._training_set_outputs)
        self._run_validation()
        