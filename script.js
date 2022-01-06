/*

NETWORK:
0  0
0  0

backprop testing


back prop layer by layer.
get "wants of each neuron"

starting from last layer:
get cost of layer:
for the first layer just use the training example.
for other layers you have the want for the layer after so
want for each neuron = weight * want for connected neuron
  for each neuron:i
    for each neuron in layer before:j
      j.activation * i.want / learning rate = weight change.

//----------

0---0



z(L) = a(L -1) * w(L) + b(L)
a(L) = sig(z(L))
y is the expected out
C = (a(L)-y)^2


C(0)   z(L)  a(L)  C(0)
---- = ----  ----  ----
w(L)   w(L)  z(L)  a(L)


C(0)
---  = 2(a(L) - y)
a(L)


a(L)
---  = σ(z(L)))⋅(1−σ(z(L))) //just the derivitave sigmoid
z(L)

z(L)
---  = a(L-1)
w(L)

a(L-1)*sigmoidPrime(z(L)) * 2(a(L) - y)
OR (idk)
a(L-1)*sigmoidPrime(z(L)) * 2(a(L-1) - y)



C(0)   z(L)  a(L)  C(0)
---- = ----  ----  ----
b(L)   b(L)  z(L)  a(L)

z(L)
---  = 1
b(L)



C(0)      z(L)     a(L)   C(0)
---    =  ---      ---    ---
a(L-1)    a(L-1)   z(L)   a(L)

NETWORK:
0  0
0  0
*/
//we want it to output the input


function sigmoid(z) {
  return 1 / (1 + Math.exp(-z));
}

function sigmoidPrime(z) {
  return Math.exp(-z) / Math.pow(1 + Math.exp(-z), 2);
}

var activationsOne = [0,0];
var activationsTwo = [0,0];
//activationsOne is the first two neurons activations.
//the network looks like this
//think of "activationsOne" as the first layer activations
/*
0    0
     
0    0
*/
//weight[0] is the first top left neuron connected to top right. weight [1] is top left connection to botom right. weight[2] is bottom left connected to top right...
var weights = [0.5,0.5,0.5,0.5];

//z is the resulting weighted sum after calculation BEFORE the sigmoid function. we store this so we can do backprop.
var z = [0,0,0,0];

var s = [0,0,0,0]; //sensitivities of each weight to the cost function


var costs = [0,0,0,0]; //per neuron
const batch_size = 5;//batch size
const learning_rate = 0.01;//learning rate
var counter = 0;// this counts the epochs basically

// var bias = [0,0];
var biasTwo = [0,0]; // The biases are useless and dont ever change. This script only backprops the weights.
var totalCost = 0;//total cost of the network.

var forward = function(data){
  activationsOne = data;
  
  z[0] = weights[0] * activationsOne[0];
  z[1] = weights[1] * activationsOne[0];
  z[2] = weights[2] * activationsOne[1];
  z[3] = weights[3] * activationsOne[1];
  
  
  activationsTwo[0] = sigmoid(z[0] + z[2] + biasTwo[0]);
  activationsTwo[1] = sigmoid(z[1] + z[3] + biasTwo[1]);
};


var train = function(input, output){
  forward(input);
  //backproping
  //first get cost of last layer
  costs[2] = Math.pow(activationsTwo[0] - output[0],2);
  costs[3] = Math.pow(activationsTwo[1] - output[1],2);
   
  totalCost = (costs[2] + costs[3]) / 2; //average of the last two neurons
  

  s[0] = activationsOne[0]*sigmoidPrime(z[0]) * 2 * (activationsTwo[0] - output[0]);
  s[1] = activationsOne[0]*sigmoidPrime(z[1]) * 2 * (activationsTwo[1] - output[1]);
  s[2] = activationsOne[1]*sigmoidPrime(z[2]) * 2 * (activationsTwo[0] - output[0]);
  s[3] = activationsOne[1]*sigmoidPrime(z[3]) * 2 * (activationsTwo[1] - output[1]);
      
//this is the algorithm to get the sensitivity of the weight to the cost function
// a(L-1)*sigmoidPrime(z(L)) * 2(a(L) - y)
  counter++;
  if(counter > batch_size){
    counter = 0;
    //apply sensitivities. //its negative cuz we want the cost to go DOWN
    for(i = 0;i<s.length;i++){
      weights[i] += -s[i]*learning_rate;
    }
    s = [0,0,0,0]; //reset the sensitivities
  }
  return totalCost;
};


for(var i = 0;i<25;i++){
  
  let x = train(
    [0.5,0.5],
    [1,1]
    );
    
    if(i % 5 === 0){
      alert(x + " " + i);
    }
}
//trains random garbage