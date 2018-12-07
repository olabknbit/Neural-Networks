# Final Project for class Neural Networks at WUT

For this project we are solving a complex problem of transportation:
Given a set of tram/bus routes, and a set of passengers' trips, how to pick a set of departure hours for the buses from their init stops.

In order to solve that problem for given routes and trips, we are going to use an approach as below:
1. For a given set of routes and trips, generate N (quite big) sets of departure hours.
2. For the generated set, calculate average trip time of all the passenger's trips.
3. Feed the data from point 1 and 2 to the neural network so that it learns to approximate the data.
4. When the accuracy is quite good (TODO - determine what good means in this scenario - will likely depend on the given params), you have a means of approximating average travel time for a lot of sets of departure hours and finding the best set. Using trained NN model will be faster than calculating accurate times using a `simulator`. 
5. Now generate even more sets of departure hours and pick the best one - the best one is the one for which average passenger's travel time is min.

In order to get the best model for a given problem, we're going to use a genetic algorithm to create a mutated perceptron.

## How to run
```bash
python final_project/run_experiments.py
```