from neural_network import NeuralNetwork
import numpy as np

class Predictor(NeuralNetwork):
    """Predictor class from Hongshan."""
    
    def __init__(self, shape=[17, 10, 5, 1], path = '', name = '', train_num = 10000):
        self.path = path
        self.name = name
        self.train_num = train_num
        NeuralNetwork.__init__(self, shape=shape)
        self.average_training_error = [] # keeps track of training error per epoch

    def _load_data(self):
        data = np.loadtxt(self.path+self.name, delimiter=',')
        bias = np.ones([data.shape[0], 1])
        data = np.append(bias, data, axis=1) # add "1" to each input to account the fact my neural net does not have bias
        return data

    def _split_the_data(self, data):
        train = data[0:self.train_num]
        test = data[self.train_num:]
        return train, test

    def _make_batch(self, data, batch_size, iteration):
        try:
            current = data[iteration:iteration+batch_size]
            features = data[:,0:-1]
            labels = data[:,-1].reshape(-1,1)
        except IndexError:
            current = data[iteration:]
            features = data[:,0:-1]
            labels = data[:,-1].reshape(-1,1)
        
        return features, labels

    def train(self, num_epoch, batch_size):
        learning_rate = 0.1
        data = self._load_data()
        train, test = self._split_the_data(data)

        for epoch in range(num_epoch):
            np.random.shuffle(train)
            num_batch = int(len(train) / batch_size)
    
            accumulated_error = 0
            for iteration in range(num_batch):
                features, labels = self._make_batch(data=train, batch_size=batch_size, iteration=iteration)
                self.forward(features)
                self.backward(labels)
                self.updateParams(learning_rate)
                accumulated_error += self.current_error
            
            average_error = accumulated_error / num_batch
            average_RMSE = np.sqrt(average_error)
            self.average_training_error.append(average_RMSE)
            print("Average RMSE of epoch {0} : {1}".format(epoch, average_RMSE))

        features = test[:,0:-1]
        ground_truth = test[:,-1].reshape(-1,1)
        self.forward(features)
        self.backward(ground_truth)
        print("Performance on the test set (RMSE): {0}".format(np.sqrt(self.current_error)))
        
        