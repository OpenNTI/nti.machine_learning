from numpy import array

from numpy.random import shuffle

from nti.data import Model
from nti.data import NTIDataFrame
from nti.data import AbstractDataSet

class SupervisedDataSet(AbstractDataSet):
    """
    Class managing a data set for use by
    a supervised learning model.
    """
    
    _indices = []
    _training_indices = []
    _validation_indices = []
    
    def __init__(self, data_frame, prediction_column, training_ratio):
        self._training_ratio = training_ratio
        self._data = data_frame
        self._prediction_column = prediction_column
        try:
            self._prediction_data = self._data[prediction_column]
            del self._data[prediction_column]
        except IndexError:
            raise ValueError("Invalid prediction column.")
        self._indices = data_frame.index.values
        shuffle(self._indices)
        training_size = int(len(self._indices) * self._training_ratio)
        self._training_indices = [self._indices[i] for i in range(training_size)]
        self._validation_indices = [self._indices[i] for i in self._indices if i not in self._training_indices]
    
    def get_training_set_inputs(self):
        """
        Get the inputs for the training set
        """
        return [self._get_from_frame(i)[0] for i in self._training_indices]
        
    def get_training_set_outputs(self):
        """
        Get the outputs for the training set
        """
        return [self._get_from_frame(i)[1] for i in self._training_indices]
    
    def get_validation_set_inputs(self):
        """
        Get the inputs for a validation set
        """
        return [self._get_from_frame(i)[0] for i in self._validation_indices]
    
    def get_validation_set_outputs(self):
        """
        Get the outputs for the validation
        """
        return [self._get_from_frame(i)[1] for i in self._validation_indices]

class SupervisedModel(Model):
    """
    A supervised learning model
    """
    
    success_rate = 0
    
    def __init__(self, data_frame, prediction_column, training_set_ratio=.7):
        if not isinstance(data_frame, NTIDataFrame):
            raise TypeError("data_frame must be of type NTIDataFrame")
        if len(data_frame) <= 1:
            raise ValueError("Insufficient data set size")
        self._data = SupervisedDataSet(data_frame, prediction_column, training_ratio=training_set_ratio)
        self._training_set_inputs = self._data.get_training_set_inputs()
        self._training_set_outputs = self._data.get_training_set_outputs()
        self._validation_set_inputs = self._data.get_validation_set_inputs()
        self._validation_set_outputs = self._data.get_validation_set_outputs()
    
    def _run_validation(self):
        """
        Run the validation set for the learning model,
        this must be implemented by the train method
        """
        aggregate = 0
        for i in range(len(self._validation_set_inputs)):
            prediction = self.classify(self._validation_set_inputs[i])
            correct = prediction == self._validation_set_outputs[i]
            if correct:
                aggregate += 1
        self.success_rate = aggregate / float(len(self._validation_set_inputs))
    
    def classify(self, inputs):
        """
        Classify a set of inputs
        """
        raise NotImplementedError("classify function not implemented")
    
    def train(self):
        """
        Train the model. Time consuming, therefore it is its own method
        that must be called rather than in the constructor.
        """
        raise NotImplementedError("train function not implemented")
        
        
