import helpers
import argparse
import numpy as np
import matplotlib.pyplot as plt



class LogisticRegression():


    def __init__(self, lr, epochs, n_classes, n_features, X_train, Y_train):

        
        self.lr = lr 
        self.history = {'cost': [], 'loss': []}
        
        self.epochs = epochs
        self.n_classes = n_classes
        self.X_train = X_train
        
        self.Y_train = Y_train.astype(int)
        self.n_examples = len(self.Y_train)
        
        self.n_features = min(self.X_train.shape)
        self.W = np.zeros((self.n_classes, self.n_features + 1))
        self.Y_encoding = self.get_one_hot_encoding(self.Y_train, self.n_classes)

        
    
    def get_one_hot_encoding(self, Y_train, n_classes):
        '''
        Get one hot encoded labels for 1 vs. all
        '''
        Y = np.full(Y_train.size*n_classes, 0).reshape(Y_train.size, n_classes)
        Y[np.arange(Y_train.size), Y_train] = 1

        return(Y)


    def add_bias(self, X):
        '''
        Takes in n_examples * n_features
        Returns (n_examples * n_features + 1) matrix
        '''
        return(np.hstack((np.ones((max(X.shape),1)),X)))


    def get_cost(self, Y_encoding, a, n_examples):
        '''
        Returns cross entropy loss
        '''
        return -np.sum(Y_encoding * np.log(a)) / n_examples



    def get_softmax(self, a):
        '''
        Takes in a vector of probabilities and 
        outputs the thresholded predictions
        '''
        return np.exp(a) / np.sum(np.exp(a), axis=1).reshape(a.shape[0], 1)


    def predict_softmax(self, X, W):
        '''
        Generate the prediction
        '''
        a = self.get_softmax(np.dot(X, W.T))
        
        return(a)


    def get_gradient_descent_step(self, a):
        '''
        Update weights according to gradient descent rule
        '''
        delta = (self.lr/self.n_examples) * np.dot((a-self.Y_encoding).T, self.X_train)
        self.W -= delta


    def train(self):
        '''
        Update weights according to gradient descent rule
        '''

        self.X_train = self.add_bias(self.X_train)

        for epoch in range(self.epochs):

            a = self.predict_softmax(self.X_train, self.W)
            cost = self.get_cost(self.Y_encoding, a, self.n_examples)
            loss = helpers.get_loss(self.Y_train, np.argmax(a, axis=1))

            W = self.get_gradient_descent_step(a) 
            self.history['cost'].append(cost)
            self.history['loss'].append(loss)
            print(cost, loss)

        return(self.history, self.W)

        


    def predict(self, X, Y):
        '''
        Predict on validation set
        '''
        X = self.add_bias(X)
        Y = Y.astype(int)

        n_val_examples = len(Y)
        preds = self.predict_softmax(X, self.W)
        preds = np.argmax(preds, axis = 1)
        loss = helpers.get_loss(Y, preds)

        return(preds, loss)


def test(lr, epochs):

    
    data_args = {

          'data_path': 'data',
          'name': 'zipcombo.dat', 
          'train_percent': 0.8,
          'k': 5,

          }

    # Load full dataset
    X, Y = helpers.load_data(data_args['data_path'], data_args['name'])
    
    # Shuffle and split dataset
    X_shuffle, Y_shuffle, perm = helpers.shuffle_data(X,Y)
    
    # Split dataset
    X_train, X_val, Y_train, Y_val, _, _ = helpers.split_data(X_shuffle, Y_shuffle, perm, data_args['train_percent'])

    # Create weak learner object
    weak_learner = LogisticRegression(lr=lr, epochs=epochs, n_classes=10, 
                                      n_features=256, X_train=X_train, Y_train=Y_train)

    # Train
    history, _ = weak_learner.train()

    # Get training predictions and loss
    train_preds, train_loss = weak_learner.predict(X_train, Y_train)

    # Evaluate
    val_preds, val_loss = weak_learner.predict(X_val, Y_val)

    return(history, train_preds, train_loss, val_preds, val_loss)



if __name__ == '__main__':

   np.random.seed(13290)

   parser = argparse.ArgumentParser(description='List the content of a folder')

   parser.add_argument('lr', type=float, 
                       help='Learning rate for weak learner....')


   parser.add_argument('epochs', type=int, help='Epochs to train weak learner for...')

   args = parser.parse_args()

   lr = args.lr

   epochs = args.epochs

   history, train_preds, train_loss, val_preds, val_loss = test(lr, epochs)

   print(history)

   helpers.get_loss_plot(history, model_name="logistic_regression", y_name='loss')
   helpers.get_loss_plot(history, model_name="logistic_regression", y_name='cost')

   print(train_loss, val_loss)