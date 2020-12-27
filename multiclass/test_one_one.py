import kernel_perceptron as perceptron
import numpy as np
import helpers
import time


def train_one_vs_one(X_train, Y_train, X_val, Y_val, n_classes):
  '''
  Train all classifiers we need for OVO
  '''
  k = 0
  d = 2
  kernel_type = 'polynomial'
  n_classifiers = 45
  histories = []
  epochs = 20
  question_no = 'test'
  classifiers_per_training = 1
  convergence_epochs = 2
  fit_type = 'one_vs_one'
  check_convergence = True

  train_predictions = np.zeros((n_classifiers, len(Y_train)))
  val_predictions = np.zeros((n_classifiers, len(Y_val)))

  train_votes = np.zeros((n_classes, len(Y_train)))
  val_votes = np.zeros((n_classes, len(Y_val)))
  
  
  for i in range(n_classes):
    for j in range(i+1, n_classes):

      X_train_sub = X_train[(Y_train == i) | (Y_train == j)]
      Y_train_sub = Y_train[(Y_train == i) | (Y_train == j)]

      X_val_sub = X_val[(Y_val == i) | (Y_val == j)]
      Y_val_sub = Y_val[(Y_val == i) | (Y_val == j)]

      settings = perceptron.train_setup(X_train_sub, Y_train_sub, X_val_sub, Y_val_sub, fit_type, 
                                        classifiers_per_training, d, kernel_type, neg = i, pos = j)

      history = perceptron.train_perceptron(*settings, X_train_sub, Y_train_sub, X_val_sub, Y_val_sub, 
                                             epochs, 
                                             classifiers_per_training, 
                                             question_no, convergence_epochs, fit_type, 
                                             check_convergence, neg=i, pos=j)

      train_predictions[k, (Y_train == i) | (Y_train == j)] = history['preds_train']
      val_predictions[k, (Y_val == i) | (Y_val == j)] = history['preds_val']

      train_neg_votes = np.zeros(len(Y_train))
      train_pos_votes = np.zeros(len(Y_train))

      val_neg_votes = np.zeros(len(Y_val))
      val_pos_votes = np.zeros(len(Y_val))

      train_neg_votes[train_predictions[k] == -1] = 1
      train_pos_votes[train_predictions[k] == 1] = 1

      val_neg_votes[val_predictions[k] == -1] = 1
      val_pos_votes[val_predictions[k] == 1] = 1
      
      train_votes[i] += train_neg_votes
      train_votes[j] += train_pos_votes

      val_votes[i] += val_neg_votes
      val_votes[j] += val_pos_votes

      histories.append(history)

      k+=1


  print((np.argmax(train_votes, axis = 0) == Y_train).sum())
  print((np.argmax(val_votes, axis = 0) == Y_val).sum())

  return(histories, train_predictions, val_predictions, train_votes, val_votes)







if __name__ == '__main__':

  train_percent = 0.8
  n_classes = 10
  X, Y = helpers.load_data('../data', 'zipcombo.dat')
  X_shuffle, Y_shuffle, perm = helpers.shuffle_data(X,Y)
  X_train, X_val, Y_train, Y_val, _, _ = helpers.split_data(X, Y, perm, train_percent)

  histories, train_predictions, val_predictions, train_votes, val_votes = train_one_vs_one(X_train, Y_train, X_val, Y_val, n_classes)

