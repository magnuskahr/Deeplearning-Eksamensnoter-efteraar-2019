# 1. Traditional Neural Networks

_Lektion 3 og 4_ 

## 0. Motivation

## 1. Lineær regression
* Explain what linear regression is (what are the inputs/outputs, and how is mapping between them described?)
* Mathematically define the model, h(x), used in linear regression.


## 2. Optimisering
* Explain conceptually what optimization is, and what is the purpose of it

### 2.1 Loss funktioner
* Explain conceptually what a loss function is, and what is the purpose of it
* Define the L2 loss function typically used in linear regression

### 2.2 Gradient descent

* Explain conceptually how gradient descent works
* Explain mathematically how gradient descent works
* Know what a partial derivative is, and explain how it is used in gradient descent
11. Know what a gradient is
12. (Eksempel) Define the partial derivate of the L2 loss used in linear regression
13. Know the difference between global and local minima of a loss function
14. Know that polynomial fitting can be implemented using linear regression

### 2.3 Learning rate

8. Explain what a learning rate is, and what is the purpose of it
9. Know what typically happens if the learning rate is set too high or too low

15. Explain conceptually the difference between overfitting and underfitting

## 3. Logistic regression

16. Explain what logistic regression is (what are the inputs/outputs, and how is mapping between them described?)
17. Mathematically define the model, h(x), used in logistic regression.
18. Know what a sigmoid function is and explain why it can be used to model probabilities for
two classes
19. Define the (cross entropy) loss function typically used in logistic regression
20. Explain how logistic regression works on images, like MNIST
21. Explain what is meant by regularization
22. Explain how weight decay works

### 3.1 Loss functions: softmax
23. Explain conceptually the difference between using L1 vs L2 regularization
24. Define the softmax function and explain how it c be used to make the model predict class
probabilities for more than two classes
25. Explain how the loss function is calculated for softmax regression – and compare with the
loss function for logistic regression
26. Explain how sofmax regression works on images, like MNIST

### 3.2 Entropy
27. Explain what entropy is (conceptually using weather station example is okay)
28. Explain what cross entropy is (conceptually using weather station example is okay)
29. Mention what KL divergence is and what it “measures”
30. Explain how cross entropy relates to logistic regression and softmax regression