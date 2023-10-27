# Work Report

## Information

- Name: <ins> DAS, NEHA </ins>
- GitHub: <ins> NehaDas25 </ins>


## Features

- Not Implemented:
  - PART 2.4: Cost Function -  was not implemented as a graded assignment. It was implemention of the cross-entropy cost function for us by the professor.
  - PART 3 - Visualizing the Word Vectors - In this part we will visualize the word vectors trained using the function we just coded in the assignment. Part 3 was already implemented by the professor. 

<br><br>

- Implemented:
  - PART 2: Training the Model
  - PART 2.1:  Initializing the Model
    - Exercise 1: initialize_model
      - Implemented a function initialize_model() that takes N,V, random_seed=1 as inputs.
      - Returns W1, W2, b1, b2: initialized weights and biases.
      - Used numpy.random.rand to generate matrices that are initialized with random values from a uniform distribution, ranging between 0 and 1.
      - This passed all the unit-test cases.

  - PART 2.2: Softmax
    - Exercise 2: softmax
      - Implemented a function softmax() that takes z as input.
      - Returns the predicted values- yhat.
      - Assumed that the input 𝑧 to softmax is a 2D array.
      - Calculated the softmax using the formula softmax(𝑧𝑖) = 𝑒^𝑧𝑖 / ∑(𝑉−1𝑖=0) 𝑒^𝑧𝑖.
      - Used numpy.exp and numpy.sum(set the axis = 0 so that it can take the sum of each column in z) to calculate yhat.
      - This passed all the unit-test cases all well.

  - PART 2.3: Forward Propagation
    - Exercise 3: forward_prop
      - Implemented a function forward_prop() that takes x, W1, W2, b1, b2 as inputs.
      - Returns h - a hidden vector and z - output score vector.
      - x:  average one hot vector for the context 
      - W1, W2, b1, b2:  matrices and biases to be learned
      - Calculated *h* using numpy.dot() with the given formula ℎ = 𝑊1𝑋 + 𝑏1.
      - Applied the activation function(ReLU) to h and store it in h using numpy.maximum() that is  h = np.maximum(0,h).
      - Calculated *z* using numpy.dot() with the given formula 𝑧 = 𝑊2ℎ + 𝑏2.
      - This passed all the unit-test cases as well.

  - PART 2.5: Training the Model - Backpropagation - CBOW MODEL 
    - Exercise 4: back_prop
      - Implemented a function back_prop() that takes x, yhat, y, h, W1, W2, b1, b2, batch_size as inputs.
      - Returns grad_W1, grad_W2, grad_b1, grad_b2:  gradients of matrices and biases.
      - x:  average one hot vector for the context 
      - yhat: prediction (estimate of y)
      - y:  target vector
      - h:  hidden vector (see eq. 1)- ℎ = 𝑊1𝑋 + 𝑏1.
      - W1, W2, b1, b2:  matrices and biases  
      - batch_size: batch size
      - Here l1 is computed using the formula given in the lecture notes that is *l1 as W2^T (Yhat - Y)* and re-use it whenever we see *W2^T (Yhat - Y)* used to compute a gradient.
      - Applied relu to l1 that is *l1 = np.maximum(0,l1)*.
      - Computed the gradient for W1 using the formula given in the lecture notes that is *[grad_W1= (1/batch_size) * np.dot(l1,x.T)]*.
      - Computed the gradient for W2 using the formula given in the lecture notes that is *[grad_W2= (1/batch_size) * np.dot((yhat-y),h.T)]*.
      - Computed the gradient for b1 using the formula given in the lecture notes that is *[grad_b1= (1/batch_size) * np.sum(l1, axis=1, keepdims=True)]*. This uses an array with all ones.
      - Computed the gradient for b2 using the formula given in the lecture notes that is *[grad_b2= (1/batch_size) * np.sum((yhat-y), axis=1, keepdims=True)]*. This uses an array with all ones as well.
      - This passed all the unit-test cases as well.

  - PART 2.6: Gradient Descent
    - Exercise 5: gradient_descent
      - Implemented a function gradient_descent() that takes data, word2Ind, N, V, num_iters, alpha=0.03,random_seed=282,initialize_model(), get_batches(), forward_prop(),softmax(), compute_cost(),back_prop(), verbose=True as inputs. 
      - Returns W1, W2, b1, b2:  updated matrices and biases after num_iters iterations.
      - data:      text
      - word2Ind:  words to Indices
      - N:         dimension of hidden vector  
      - V:         dimension of vocabulary 
      - num_iters: number of iterations  
      - random_seed: random seed to initialize the model's matrices and vectors
      - initialize_model: implementated the function to initialize the model
      - get_batches: helper function to get the data in batches
      - forward_prop: implementated the function to perform forward propagation
      - softmax: implementated the softmax function
      - compute_cost: cost function (Cross entropy)
      - back_prop: implementated the function to perform backward propagation
      - Used *initialize_model(N,V, random_seed=random_seed)* to generate matrices that are initialized with random values from a uniform distribution, ranging between 0 and 1.
      - Used the helper function *get_batches(data, word2Ind, V, C, batch_size)*.
      - Used *forward_prop(x, W1, W2, b1, b2)* to get the z and h.
      - Used *softmax(z)* to get yhat.
      - Used *compute_cost(y, yhat, batch_size)* to get cost of Entropy.
      - Used *back_prop(x, yhat, y, h, W1, W2, b1, b2, batch_size)* to get gradients grad_W1, grad_W2, grad_b1, grad_b2.
      - Update the weights and biases using alpha for example *W1 = W1 - alpha * grad_W1.* The same goes for W2, b1, b2.
      - This passed all the unit-test cases as well.

      
<br><br>

- Partly implemented:
  - utils2.py which contains sigmoid(z), get_idx(words, word2Ind), pack_idx_with_frequency(context_words, word2Ind), get_vectors(data, word2Ind, V, C), get_batches(data, word2Ind, V, C, batch_size), compute_pca(data, n_components=2), get_dict(data),  get_emoji_regexp() has not been implemented, it was provided.
  - w4_unittest.py was also not implemented as part of assignment to pass all unit-tests for the graded functions().
  - support_files was also not implemented as part of assignment to pass all unit-tests for the graded functions().

<br><br>

- Bugs
  - No bugs

<br><br>


## Reflections

- Assignment is very good. Gives a thorough understanding of the basis of Word Embedding, CBOW Model, Activation function, Forward Propagation, Gradient Descent, Backward Propagation, Entropy loss.


## Output

### output:

<pre>
<br/><br/>
Out[1] - True

Out[3] - Number of tokens: 60976 
['o', 'for', 'a', 'muse', 'of', 'fire', '.', 'that', 'would', 'ascend', 'the', 'brightest', 'heaven', 'of', 'invention']
 
Out[4] - 

Size of vocabulary:  5775
Most frequent tokens:  [('.', 9630), ('the', 1521), ('and', 1394), ('i', 1257), ('to', 1159), ('of', 1093), ('my', 857), ('that', 781), ('in', 770), ('a', 752), ('you', 748), ('is', 630), ('not', 559), ('for', 467), ('it', 460), ('with', 441), ('his', 434), ('but', 417), ('me', 417), ('your', 397)]

Out[5] -  Size of vocabulary:  5775

Out[6] -  
Index of the word 'king' :   2744
Word which has index 2743:   kinds

Out[8] - 

tmp_W1.shape: (4, 10)
tmp_W2.shape: (10, 4)
tmp_b1.shape: (4, 1)
tmp_b2.shape: (10, 1)

Expected Output
tmp_W1.shape: (4, 10)
tmp_W2.shape: (10, 4)
tmp_b1.shape: (4, 1)
tmp_b2.shape: (10, 1)

Out[9] -  All tests passed

Out[11] -
array([[0.5       , 0.73105858, 0.88079708],
       [0.5       , 0.26894142, 0.11920292]])

Expected Ouput
array([[0.5       , 0.73105858, 0.88079708],
       [0.5       , 0.26894142, 0.11920292]])

Out[12] - All tests passed

Out[14] -

x has shape (3, 1)
N is 2 and vocabulary size V is 3
call forward_prop

z has shape (3, 1)
z has values:
[[0.55379268]
 [1.58960774]
 [1.50722933]]

h has shape (2, 1)
h has values:
[[0.92477674]
 [1.02487333]]

Expected output
x has shape (3, 1)
N is 2 and vocabulary size V is 3
call forward_prop

z has shape (3, 1)
z has values:
[[0.55379268]
 [1.58960774]
 [1.50722933]]

h has shape (2, 1)
h has values:
[[0.92477674]
 [1.02487333]]

Out[15] - All tests passed

Out[17] -

tmp_x.shape (5775, 4)
tmp_y.shape (5775, 4)
tmp_W1.shape (50, 5775)
tmp_W2.shape (5775, 50)
tmp_b1.shape (50, 1)
tmp_b2.shape (5775, 1)
tmp_z.shape: (5775, 4)
tmp_h.shape: (50, 4)
tmp_yhat.shape: (5775, 4)
call compute_cost
tmp_cost 10.4074

Expected output
tmp_x.shape (5775, 4)
tmp_y.shape (5775, 4)
tmp_W1.shape (50, 5775)
tmp_W2.shape (5775, 50)
tmp_b1.shape (50, 1)
tmp_b2.shape (5775, 1)
tmp_z.shape: (5775, 4)
tmp_h.shape: (50, 4)
tmp_yhat.shape: (5775, 4)
call compute_cost
tmp_cost 10.4074 

Out[19] - 

get a batch of data
tmp_x.shape (5775, 4)
tmp_y.shape (5775, 4)

Initialize weights and biases
tmp_W1.shape (50, 5775)
tmp_W2.shape (5775, 50)
tmp_b1.shape (50, 1)
tmp_b2.shape (5775, 1)

Forward prop to get z and h
tmp_z.shape: (5775, 4)
tmp_h.shape: (50, 4)

Get yhat by calling softmax
tmp_yhat.shape: (5775, 4)

call back_prop
tmp_grad_W1.shape (50, 5775)
tmp_grad_W2.shape (5775, 50)
tmp_grad_b1.shape (50, 1)
tmp_grad_b2.shape (5775, 1)

Expected output
get a batch of data
tmp_x.shape (5775, 4)
tmp_y.shape (5775, 4)

Initialize weights and biases
tmp_W1.shape (50, 5775)
tmp_W2.shape (5775, 50)
tmp_b1.shape (50, 1)
tmp_b2.shape (5775, 1)

Forwad prop to get z and h
tmp_z.shape: (5775, 4)
tmp_h.shape: (50, 4)

Get yhat by calling softmax
tmp_yhat.shape: (5775, 4)

call back_prop
tmp_grad_W1.shape (50, 5775)
tmp_grad_W2.shape (5775, 50)
tmp_grad_b1.shape (50, 1)
tmp_grad_b2.shape (5775, 1)
 
Out[20] -  All tests passed

Out[22] -

Call gradient_descent
iters: 10 cost: 8.538367
iters: 20 cost: 4.449100
iters: 30 cost: 16.154438
iters: 40 cost: 2.372795
iters: 50 cost: 10.508012
iters: 60 cost: 7.730859
iters: 70 cost: 5.893077
iters: 80 cost: 9.354901
iters: 90 cost: 10.002799
iters: 100 cost: 11.484674
iters: 110 cost: 4.625150
iters: 120 cost: 4.428295
iters: 130 cost: 10.306100
iters: 140 cost: 6.705970
iters: 150 cost: 3.189159

Expected Output
iters: 10 cost: 11.714748
iters: 20 cost: 3.788280
iters: 30 cost: 9.179923
iters: 40 cost: 1.747809
iters: 50 cost: 8.706968
iters: 60 cost: 10.182652
iters: 70 cost: 7.258762
iters: 80 cost: 10.214489
iters: 90 cost: 9.311061
iters: 100 cost: 10.103939
iters: 110 cost: 5.582018
iters: 120 cost: 4.330974
iters: 130 cost: 9.436612
iters: 140 cost: 6.875775
iters: 150 cost: 2.874090

  **Note - Your numbers may differ a bit depending on which version of Python you're using** 

Out[23] - All tests passed

Out[24] -

(10, 50) [2744, 3949, 2960, 3022, 5672, 1452, 5671, 4189, 2315, 4276]

Out[25] -

Out[26] -
 
 

<br/><br/>
</pre>
