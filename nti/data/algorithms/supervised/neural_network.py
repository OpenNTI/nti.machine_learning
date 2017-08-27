import random
import tensorflow as tf

from nti.data.algorithms.supervised import SampleDataSet
from nti.data.algorithms.supervised import point_list_to_dataset

class _HiddenLayer():
    
    def __init__(self, input, output):
        self.W = tf.Variable(tf.random_normal([input, output], stddev=0.03))
        self.b = tf.Variable(tf.random_normal([output], stddev=0.03))
        
    def set_output(self, input):
        self.temp = out = tf.add(tf.matmul(input, self.W), self.b)
        self.out = tf.nn.relu(out)
        
    def __repr__(self):
        return "W: %s    b: %s" % (self.W, self.b)

class NeuralNetwork():
    
    def __init__(self, x_s, y_s, layers, training_size = .5):
        if x_s is None or y_s is None or len(x_s) <= 0:
            raise ValueError("Must have at least one point.")
        
        
        self.data_set = SampleDataSet(x_s, y_s)
        self.training_size = training_size
        self.success_rate = 0.0
        
        
        self.input = tf.placeholder(tf.float32, [None, layers[0]], name="input")
        self.output = tf.placeholder(tf.float32, [None, layers[-1]], name="output")
        
        W1 = tf.Variable(tf.random_normal([layers[0], layers[1]], stddev=0.03), name='W1')
        b1 = tf.Variable(tf.random_normal([layers[1]]), name='b1')
        
        W2 = tf.Variable(tf.random_normal([layers[1], layers[2]], stddev=0.03), name='W2')
        b2 = tf.Variable(tf.random_normal([layers[2]]), name='b2')
        
        hidden_out = tf.add(tf.matmul(self.input, W1), b1)
        hidden_out = tf.nn.relu(hidden_out)
        
#         self.layers = []
#         for i in range(1, len(layers)):
#             new_layer = _HiddenLayer(layers[i-1], layers[i])
#             self.layers.append(new_layer)
#         self.layers[0].set_output(self.input)
#         for i in range(1, len(self.layers)):
#             self.layers[i].set_output(self.layers[i-1].out)
        
        #self.y_ = tf.nn.softmax(self.layers[-1].out)
        self.y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W2), b2))
        y_clipped = tf.clip_by_value(self.y_, 1e-10, 0.9999999)
        self.cross_entropy = -tf.reduce_mean(tf.reduce_sum(self.output * tf.log(y_clipped)
                         + (1 - self.output) * tf.log(1 - y_clipped), axis=1))
        self.optimiser = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(self.cross_entropy)
        self.init_op = tf.global_variables_initializer()
        
        self.correct_prediction = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
    
    def _randomize(self, x, y):
        indxs = [i for i in range(len(x))]
        random.shuffle(indxs)
        result_x = [x[i] for i in indxs]
        result_y = [y[i] for i in indxs]
        return result_x, result_y
    
    def train(self, epochs):
        validation_set_size = int(float(self.data_set.get_size() * (1.0 - self.training_size)))
        validation_set = point_list_to_dataset([self.data_set.get_point(i) for i in range(validation_set_size)])
        training_set = point_list_to_dataset([p for p in self.data_set if p not in validation_set])
        
        training_X = training_set.get_X_as_numpy()
        training_Y = training_set.get_Y_as_numpy(rank=True)
        
        val_X = validation_set.get_X_as_numpy()
        val_Y = validation_set.get_Y_as_numpy(rank=True)
        
        with tf.Session() as session:
            session.run(self.init_op)
            for e in range(epochs):
                avg_cost = 0
                for i in range(len(training_X)):
                    _, c = session.run([self.optimiser, self.cross_entropy], feed_dict={self.input: [training_X[i]], 
                                                                                        self.output: [training_Y[i]]})
                    avg_cost += c / len(training_X)
                print(avg_cost)
                training_X, training_Y = self._randomize(training_X, training_Y)
            for i in range(len(val_X)):
                self.success_rate = session.run(self.accuracy, feed_dict={self.input: [val_X[i]], self.output: [val_Y[i]]}) * 100
            print(self.success_rate)
            
            
    def classify(self, inputs):
        if not isinstance(inputs, list):
            raise ValueError("Inputs must be a list")
        with tf.Session() as session:
            session.run(self.init_op)
            print(session.run(self.y_, {self.input: [inputs]}))
        