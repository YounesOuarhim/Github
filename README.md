## Usage of the code
The goal of this 
This code allows us to make the optimization of the code of the hierarchical functions provided by Miss [Alix Chazottes](https://linkedin.com/in/alix-chazottes) used in image classification. There is also a performance comparison between the different loss functions used. 

## Code execution 

### Visualizing optimization results : 

To replicate the method used for code optimization. First install Scalene with PiP, then in the terminal and being in the folder where all three Python programs are stored write the following command scalene --html -outfile optimization.html cifar.py True False 0.01 1 "['automobile', 'cat', 'dog', 'horse', 'truck']" "[None, 0, 0, 1, 2, 2, 2, 1]". 
The two pages giving the optimization results are given and are named computation_time_before_optimization.html computation_time_after_optimization.html. The result from CProfile (profiling_result.txt) is also given to explain the intensive operations that the hierarchical loss function implements, without appearing in the code. 


### Comparaison results of the three Loss functions : 

master_train.py should be executed in scripts. Tensorboard should also be openned to visualize the evolutions of the loss functions of the training.





