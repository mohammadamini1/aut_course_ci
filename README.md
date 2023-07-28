# aut_course_ci

Computational Intelligence course projects

Project 1:
In this project, the goal is to use feedforward fully connected neural networks to classify fruits. The dataset used is "Fruit-360," focusing on four fruit classes. Feature extraction techniques reduce the input dimensionality, and the output layer has four neurons representing the fruit classes. The project involves loading data, implementing feedforward and backpropagation with vectorization for efficiency, and evaluating model accuracy on training and testing datasets.

Project 2:
This project involves solving the inverted pendulum control problem using fuzzy logic. The objective is to stabilize the pendulum attached to a cart by applying the appropriate force to the cart. The inputs include the pendulum angle, angular velocity, cart position, and cart velocity, which will be fuzzified into fuzzy sets. Fuzzy rules will be used to determine the output force based on the activated fuzzy sets. Finally, defuzzification will convert the fuzzy output sets into a crisp force value, aiming to keep the inverted pendulum balanced in an upright position for as long as possible.

Project 3:
This project focuses on applying evolutionary algorithms to train a neural network in a data-scarce environment, specifically a game called "Snail Jumper." The project involves implementing a neural network with specified architecture and activation functions to make decisions in the game. During the evolutionary phase, a population of players with neural networks as their brains is generated, and their performance is evaluated based on the distance covered in the game. The fittest players are selected through methods like roulette wheel or tournament selection, and new offspring are generated using crossover and mutation operations. The goal is to evolve a population of players over generations, improving their performance without explicit training data or backpropagation.
