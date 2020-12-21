def one_vs_one(X_train, Y_train, X_test):
  '''
  Input:
  Output:
  '''

  n_samples = len(X_train)
  
  predictions = np.array([np.empty(n)])
  
  votes = np.zeros((n_classes, len(test_data)))
  
  for i in range(3):
    for j in range(i+1, 3):
      
      data = training_data[(training_labels == i) | (training_labels == j)]
      j_labels = training_labels[(training_labels == i) | (training_labels == j)]
      
      labels = np.ones(len(j_labels), np.int32)
      labels[j_labels == j] = -1

      weights = perceptron(data, labels)
      prediction = test_data.dot(weights)

      i_votes = np.zeros(len(test_data))
      i_votes[prediction > 0] = 1
      
      votes[i] += i_votes
      j_votes = np.zeros(len(test_data))
      j_votes[prediction <= 0] = 1
      
      votes[j] += j_votes
  
  return np.argmax(votes, axis=0)