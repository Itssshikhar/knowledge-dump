#backpropagation #topologies #classification #natural_selection 

## References
- https://blog.otoro.net/2016/05/07/backprop-neat/

Generally, NEAT works by mutation and crossover over a network that changes it's structural complexity and initializing random weights. But given the power of backpropagation in Neural Network these days, it is better to use it to set the weights of the network based on taking the derivative of an objective function, leading to a gradient descent, hence minimizing the overall loss.

To generate a picture, the coordinates of every pixel in that picture will be the inputs to the neural network, and the outputs will be the colour for that very pixel. The more complicated the network, the more detailed the resulting picture would be.

In implementation of NEAT, there are many types of activation functions, represented as different colors in a neural network. 

![[Pasted image 20240907155819.png]]

The `add` operator does nothing to the input (which is the weighted sum of outputs of incoming connections), while the `mult` operator would multiple all the weighted inputs together. By allowing a sinusoidal operator, the network can produce repetitive patterns in the output. The `square` and `abs` operators are useful for generating symmetries. `Gaussian` operator can be helpful to draw one-off clustered regions.

Since the NEAT implementation works by describing a neural network that is ultimately built as a computation graph that can be processed by `recurrent.js`, we will also be able perform back propagation and optimise the weights of each individual neural network architecture to best fit the training data. This way, **NEAT is strictly responsible for figuring out new architectures, while backprop can try to determine the best set of weights for each architecture that NEAT comes up with**. 

In the original NEAT paper, NEAT is also used to figure out the weights as well using the genetic algorithm operations but I think this is not efficient at all, especially when we know backprop is a superior method for figuring out weights for a simple classification problem. In my implementation, I also incorporate a `L2 regularisation` term for the weights when performing backprop on each network.

When evaluating each network, we need to assign each network with a ==fitness score== on how good it is, so we can rank them in the genetic algorithm. In addition to seeing how well each network fits the training data, using [maximum likelihood](https://en.wikipedia.org/wiki/Logistic_regression#Maximum_likelihood_estimation), the number of connections would also affect the fitness score of the network. We would **prefer a simpler network over a more complicated network**, if they achieve the same regression accuracy, and in some cases we would even prefer a much simpler network over a very complicated one even if the simpler one fits the training data less accurately. 

To achieve this, multiply the fitness score by a factor that grows in proportion to the square root of the number of connections.

$fitness= −regression_​​​error ∗ √​(1+connection_count_penalty∗ connection_count)​​​$

![[Pasted image 20240907163441.png]]

One of the more interesting things I noticed is that networks that `backprop` well will tend to be favoured in the evolution process, compared to nasty networks with gradients that blow up, since a network with blown up weight values will likely give rubbish classification results that will result in a poor fitness score. Setting a low number of `backprop steps`, or a `large learning rate`, may lead the genetic algorithm to produce different types of network that will perform better in those environments.

Kind of like saying that, if even if a network can fit on the training data perfectly, if the backprop will not allow that, the network would be discarded in the evolution. Just like in real world, where let's say an Intelligent person couldn't show their full potential due to being born in a barbaric era or in modern times, a super Intelligent person couldn't convince other peers of his ideas because he lacks people skills.