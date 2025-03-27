#topologies #evolution #algorithm
## References
- https://macwha.medium.com/evolving-ais-using-a-neat-algorithm-2d154c623828
- https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf (OG paper)

## Algorithm

NEAT is an example of a topology and weight evolving artificial neural network (TWEANN), in the class of [[Genetic Algorithms]], which attempts to simultaneously learn weight values and an appropriate topology for a neural network.

The NEAT approach begins with a [perceptron](https://en.wikipedia.org/wiki/Perceptron "Perceptron")-like feed-forward network of only input neurons and output neurons. As evolution progresses through discrete steps, the complexity of the network's topology may grow, either by inserting a new neuron into a connection path, or by creating a new connection between (formerly unconnected) neurons.

if a genome contains neurons _A_, _B_ and _C_ and is represented by [A B C], if this genome is crossed with an identical genome (in terms of functionality) but ordered [C B A] crossover will yield children that are missing information ([A B A] or [C B C]), in fact 1/3 of the information has been lost in this example. NEAT solves this problem by tracking the history of genes by the use of a global innovation number which increases as new genes are added. When adding a new gene the global innovation number is incremented and assigned to that gene. Thus the higher the number the more recently the gene was added. For a particular generation if an identical mutation occurs in more than one genome they are both given the same number, beyond that however the mutation number will remain unchanged indefinitely.

These innovation numbers allow NEAT to match up genes which can be crossed with each other.

## How it works

![[1_K1bRLDs7h1tmEB4qZCLpxw.webp]]  

NEAT networks works by having limited nodes at the beginning itself and that number itself doesn't change. In the above image, 1,2 and 3 node are Input Nodes and the output Node is 4. The 5th Node is added to instantiate changes in the network (==called as Mutation==), something like back-propagation as without it, the network would just remain the same.

But adding a Node changes the structure of Network. So what actually happens is that the weight given is changed, kind of like adding a bias, so that the network would change without adding more complexity to it.

Also, the Innovation Number are the most important unique identifier for each connection. If the same innovation exists, between two networks, it can be said that the networks are ***structurally identical***.

### Mutation

The NEAT algorithm handles this process a little differently, using a process very similar to natural evolution — random mutations.

-  Add a connection between two unconnected nodes.
-  Add a node between two nodes that are already connected, disabling the old   connection & creating two new ones to preserve the structure of the network.
- Change the weight of a connection.

![[1_vtQ7OH3iMT6OmPyeRNcnYg.webp]]

Since, the mutation is random, it is very hard to that the mutation will help in changing the network. That is why, it is generally avoided to do the above point 1 and 2.

-  When adding a connection we can normalize the weight to be around 0, meaning that it will transfer very little information.
-  When adding a node we can set the weight of the first connection to that of the  previous connection and the weight of the second connection to 1. This means that the new node won’t change the output, only the structure.

## Crossovers

![[1_NCmCqKxG4Fw2Ba9wD8jP4g.webp]]

So, what would we do when the networks aren't identical structurally? This is where Crossover comes in. Did you remember the Innovation Number from above? Yup, it's gonna help us. To do this, there are 3 things to keep in mind:

- 1. First of all, we need to select a dominant parent. This is the parent that will specify the structure of the child network.
- 2. Secondly, we find all of the connections shared by both parents. You’d think that this would be difficult, but since we have innovation numbers, we can just take each connection whose innovation number appears in both parent networks.
- 3. For any connection shared by both parents, we randomly give the child one of the connections. The ensures that the weights of any shared connection in the child network will randomly come from either parent.
- 4. Finally, for any connections not shared by both networks, the child simply inherits them from the dominant parent.

After this process, the offspring would bear characteristics from both parents.

## Speciation

Now, because we are avoiding random mutations that would change the network structurally and make it worse, we really don't want that but also, how would the network then ***Mutate***?! "That's paradoxical, but it works", famous quote from a certain movie.

Here comes our hero, Speciation, to save us. Speciation tells us that the network is judged on based on the performance of similar network, not just to it's own. It's like how, at times, Virat Kohli's form is unmatched but some days it's not and that's perfectly fine. Similarly, the performance of the network is taken as a team sport.

Speciation splits up the population into several species or a teams, based on how similar they are to each other. How do we calculate their similarity? Well, innovation numbers are here to help again! We can just count how many connections they have that share innovation numbers and divide that by how many connections don’t share innovation numbers. Then, when we’re assigning a score to a network, we consider the score of the entire species, not just the individual network. This helps networks with new mutations to receive acceptable scores so that they can reproduce and mutate, slowly optimizing their mutations.

## How does it come together?

The parallels between NEAT and evolution don’t stop with their dynamic structure — the very way that we train these networks is also very similar to natural selection.

1. Firstly we need a blank population of networks. Each of these networks will initially only have the input and output nodes — no hidden nodes or connections.
2. We then randomly mutate this population. Add a connection here, mess with the weights there, so on and so forth.
3. Now it’s time to evaluate each of the networks. Simulate your environment and test each of the networks individually, recording how well each one does.
4. Time to start evolving! Throw out half of your population, namely the ones that didn’t do so well in the evaluation, and generate a new population with the well-performing half using the speciation, crossover and mutation techniques aligned above.

Then just repeat the simulation step until you’re satisfied with their performance! Over time the networks will learn to score better and better on your evaluation, by proxy learning to do exactly what you want.