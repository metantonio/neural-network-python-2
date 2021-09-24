from neural_network import NeuralNetwork
from formulae import calculate_average_error, seed_random_number_generator
from video import generate_writer, annotate_frame, take_still
#import main
from configurationNetwork import examples, new_situation, training_iterations, neurons_in_layers, show_iterations, video_file_name
import configurationNetwork

#class TrainingExample():
    #def __init__(self, inputs, output):
        #self.inputs = inputs
        #self.output = output



# Seed the random number generator
seed_random_number_generator()

# Assemble a neural network, with 3 neurons (by default) in the first layer or INPUT LAYER
# 4 neurons in the second layer or HIDDEN LAYER and 1 neuron in the third layer or OUTPUT LAYER
network = NeuralNetwork(configurationNetwork.neurons_in_layers)

# Training set with inputs [a,b,c] and output
#examples = [TrainingExample([0, 0, 1], 0),
            #TrainingExample([0, 1, 1], 1),
            #TrainingExample([1, 0, 1], 1),
            #TrainingExample([1, 1, 1], 1)]

# Create a video and image writer
fig, writer = generate_writer()

# Generate an image of the neural network before training
print ("Generating an image of the neural network before")
network.do_not_think()
network.draw()
#take_still("neural_network_before.png")

# Generate a video of the neural network learning
print ("Generating a video of the neural network learning.")
print ("There will be " + str(len(configurationNetwork.examples) * len(configurationNetwork.show_iterations)) + " frames.")
print ("This may take a long time. Please wait...")
with writer.saving(fig, configurationNetwork.video_file_name, 100):
    for i in range(1, configurationNetwork.training_iterations + 1):
        cumulative_error = 0
        for e, example in enumerate(configurationNetwork.examples):
            cumulative_error += network.train(example)
            if i in configurationNetwork.show_iterations:
                network.draw()
                annotate_frame(i, e, average_error, example)
                writer.grab_frame()
        average_error = calculate_average_error(cumulative_error, len(configurationNetwork.examples))
print ("Success! Open the file " + configurationNetwork.video_file_name + " to view the video.")

# Generate an image of the neural network after training
print ("Generating an image of the neural network after")
network.do_not_think()
network.draw()
#take_still("neural_network_after.png")

# Consider a new situation
new_situation = [1, 0, 1]
print ("Considering a new situation " + str(configurationNetwork.new_situation) + "?")
print("output estimated is: ")
print (network.think(configurationNetwork.new_situation))
network.draw()
#take_still("neural_network_new_situation.png")

