<!-- --- -->
# layout: post
# title: "What I learnt by Building a Neural Net from Scratch"
# date: 2025-07-26
# category: Machine Learning
# tags: ["Neural Nets", "Building from Scratch"]
# ---

So I have been working on building a neural net from scratch in the couple of weeks and I have learnt a lot of new things.  
I used to think that building a Neural net is pretty hard but it's not actually that hard if you really sit down and scratch your head a lil bit you will eventually build one just like me.  
At first I thought, "this is it I have built it" but then some error comes or outputs are not as you expect or training takes a lot of time and all those small problems arise from somewhere. Then you find yourself in Problem - Solve loop where you are basically solving the problems that are coming after solving the previous one.  
Basic structure of a neural net is simple: Input -> Layers -> Output. Structure of processes: Input -> Forward prop -> Compute cost or store it -> Backward prop -> Gradient Descent -> Repeat until epochs are compelte or you reach your goal.  
# BUILDING THE BASIC ELEMENTS  
In this section we will build all the basic elements required to build a neural net from scratch. This includes: Forward prop, Backward prop, Train and a Prdict function, A perameter initialization function.  
Here is a simple analogy to think of Neural Nets. You can think of a neural network as something like a factory where you put raw potatoes (data) in and chips (output) comes out. It's obvious that those potatoes have gone through a lot of hidden processing in the factory (neural net). And we go through this cycle (forward prop) again and again until we produce the chips that are one of the best. To produce the best chips we need a standard to compare themselves against and that standard is the best chips in the city (output values of input matrix, y).
## Forward Propagation  
Forward prop and forward propagation are the same if you are confused about that.  
So, why do we need forward prop? The reason is simple. You want your model to learn from the underlyning data patterns, each layer of the neural network learns something different about the data. That's why you pass the data ahead in the network.   
The output layer of the network generally outputs the type of output (generally vectors and matrices) that matches the output labels with the training data, so that computing the accuracy of our predictions becomes easy, and this is the right way to do it.  
In this section I am going to specifically talk about the forward pass for neural networks.  
Forward pass in case of Neural Networks is quite different than that of some simple linear model, when you do forward pass in the neural network you go on storing activation values and the outputs of different layers in cache (some kind of memory, don't know much about it now :p).  
When I first learnt about it I though why we are even doing this why it's not that simple, but since we are building a neural net so we can expect such hard things to come along our way.  
As the data flows ahead in the neural net, each layer gives it's output which is based on the input from the previous layer. In this sense first layer gets it's inputs from the input matrix itself (also called input layer or layer 0).  
Each layer computes Z's  and  activation values, and we are going to store them as we will need them for backprop. We usually store them in a dictionary, I am doing the same in my project.  
We are not done when the output comes out of the network because we are not sure if it is the right output or the model is learning the right things or if the perameters model has learnt are right. We need to improve our perameters as our output is the result of those perameters.  
To make sure how much accurate we are, we compute a loss function. The simple idea behind computing a loss function is to know how much total error our predictions have.  
The loss function evaluates how far the model's predictions are from the true values. We compute the loss for each example and average them to guide the model’s learning. 
There are a lot of loss functions depending upon the task you are into, some of the common ones are: mean squared loss, binary cross entropy, cross entropy etc.  
- **Mean Squared Loss**: This is a simple cost function. The idea is that you plot your predictions and true values on a plane, find the difference between predicted value and true value for some input, square this difference and then take average of that. That is why it is "mean" of losses that are squared. {HELPING GRAPHIC AND FORMULA  img here}  
- **Binary Cross Entropy**: This function is used if you are dealing with the tasks where you need to predict out of two available classes which class the example belongs to, that just means for binary classification tasks. It is generally used with sigmoid which is also used for binary classification tasks. 
$$
\text{BCE}(y, \hat{y}) = - \left[ y \cdot \log(\hat{y}) + (1 - y) \cdot \log(1 - \hat{y}) \right]
$$

