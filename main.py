import matplotlib
from matplotlib import pyplot, animation, rcParams
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import cos, sin, atan
from formulae import calculate_average_error, seed_random_number_generator
from formulae import sigmoid, sigmoid_derivative, random_weight, get_synapse_colour, adjust_line_to_perimeter_of_circle, layer_left_margin
import os
import configurationNetwork
from configurationNetwork import examples, new_situation, training_iterations, neurons_in_layers, show_iterations

class Synapse():
    def __init__(self, input_neuron_index, x1, x2, y1, y2):
        self.input_neuron_index = input_neuron_index
        self.weight = random_weight()
        self.signal = 0
        x1, x2, y1, y2 = adjust_line_to_perimeter_of_circle(x1, x2, y1, y2)
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def draw(self):
        line = pyplot.Line2D((self.x1, self.x2), (self.y1, self.y2), lw=fabs(self.weight), color=get_synapse_colour(self.weight), zorder=1)
        outer_glow = pyplot.Line2D((self.x1, self.x2), (self.y1, self.y2), lw=(fabs(self.weight) * 2), color=get_synapse_colour(self.weight), zorder=2, alpha=self.signal * 0.4)
        pyplot.gca().add_line(line)
        pyplot.gca().add_line(outer_glow)


class Neuron():
    def __init__(self, x, y, previous_layer):
        self.x = x
        self.y = y
        self.output = 0
        self.synapses = []
        self.error = 0
        index = 0
        if previous_layer:
            for input_neuron in previous_layer.neurons:
                synapse = Synapse(index, x, input_neuron.x, y, input_neuron.y)
                self.synapses.append(synapse)
                index += 1

    def train(self, previous_layer):
        for synapse in self.synapses:
            # Propagate the error back down the synapse to the neuron in the layer below
            previous_layer.neurons[synapse.input_neuron_index].error += self.error * sigmoid_derivative(self.output) * synapse.weight
            # Adjust the synapse weight
            synapse.weight += synapse.signal * self.error * sigmoid_derivative(self.output)
        return previous_layer

    def think(self, previous_layer):
        activity = 0
        for synapse in self.synapses:
            synapse.signal = previous_layer.neurons[synapse.input_neuron_index].output
            activity += synapse.weight * synapse.signal
        self.output = sigmoid(activity)

    
    def draw(self, neuron_radius):
        circle = pyplot.Circle((self.x, self.y), radius=neuron_radius, fill=False)
        pyplot.gca().add_patch(circle)


class Layer():
    def __init__(self, network, number_of_neurons, number_of_neurons_in_widest_layer):
        self.vertical_distance_between_layers = 6
        self.horizontal_distance_between_neurons = 2
        self.neuron_radius = 0.5
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.previous_layer = self.__get_previous_layer(network)
        self.y = self.__calculate_layer_y_position()
        self.neurons = self.__intialise_neurons(number_of_neurons)

    def __intialise_neurons(self, number_of_neurons):
        neurons = []
        x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        for iteration in range(number_of_neurons):
            neuron = Neuron(x, self.y, self.previous_layer)
            neurons.append(neuron)
            x += self.horizontal_distance_between_neurons
        return neurons

    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        return self.horizontal_distance_between_neurons * (self.number_of_neurons_in_widest_layer - number_of_neurons) / 2

    def __calculate_layer_y_position(self):
        if self.previous_layer:
            return self.previous_layer.y + self.vertical_distance_between_layers
        else:
            return 0

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def __line_between_two_neurons(self, neuron1, neuron2):
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = self.neuron_radius * sin(angle)
        y_adjustment = self.neuron_radius * cos(angle)
        line = pyplot.Line2D((neuron1.x - x_adjustment, neuron2.x + x_adjustment), (neuron1.y - y_adjustment, neuron2.y + y_adjustment))
        pyplot.gca().add_line(line)

    def think(self):
        for neuron in self.neurons:
            neuron.think(self.previous_layer)

    def draw(self, layerType=0):
        for neuron in self.neurons:
            neuron.draw( self.neuron_radius )
            if self.previous_layer:
                for previous_layer_neuron in self.previous_layer.neurons:
                    self.__line_between_two_neurons(neuron, previous_layer_neuron)
        # write Text
        x_text = self.number_of_neurons_in_widest_layer * self.horizontal_distance_between_neurons
        if layerType == 0:
            pyplot.text(x_text, self.y, 'Input Layer', fontsize = 12)
        elif layerType == -1:
            pyplot.text(x_text, self.y, 'Output Layer', fontsize = 12)
        else:
            pyplot.text(x_text, self.y, 'Hidden Layer '+str(layerType), fontsize = 12)

class NeuralNetwork():
    def __init__(self, number_of_neurons_in_widest_layer):
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.layers = []
        self.layertype = 0

    def add_layer(self, number_of_neurons ):
        layer = Layer(self, number_of_neurons, self.number_of_neurons_in_widest_layer)
        self.layers.append(layer)

    def draw(self):
        pyplot.figure()
        for i in range( len(self.layers) ):
            layer = self.layers[i]
            if i == len(self.layers)-1:
                i = -1
            layer.draw( i )
        pyplot.axis('scaled')
        pyplot.axis('off')
        pyplot.title( 'Neural Network architecture', fontsize=15 )
        pyplot.savefig("Neural Network architecture.jpg")
        pyplot.show()
        #pyplot.savefig("Neural Network architecture.jpg")

    def train(self, example):
        error = example.output - self.think(example.inputs)
        self.reset_errors()
        self.layers[-1].neurons[0].error = error
        for l in range(len(self.layers) - 1, 0, -1):
            for neuron in self.layers[l].neurons:
                self.layers[l - 1] = neuron.train(self.layers[l - 1])
        return fabs(error)

    def do_not_think(self):
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.output = 0
                for synapse in neuron.synapses:
                    synapse.signal = 0

    def think(self, inputs):
        for layer in self.layers:
            if layer.is_input_layer:
                for index, value in enumerate(inputs):
                    self.layers[0].neurons[index].output = value
            else:
                layer.think()
        return self.layers[-1].neurons[0].output

    def reset_errors(self):
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.error = 0

class DrawNN():
    def __init__( self, neural_network ):
        self.neural_network = neural_network

    def draw( self ):
        widest_layer = max( self.neural_network )
        network = NeuralNetwork( widest_layer )
        for l in self.neural_network:
            network.add_layer(l)
        network.draw()

def take_still(image_file_name):
    pyplot.savefig(image_file_name)

def generate_writer():
    FFMpegWriter = animation.writers['ffmpeg']
    writer = FFMpegWriter(fps=frames_per_second, metadata=metadata)
    fig = pyplot.figure()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    pyplot.xlim(0, width)
    pyplot.ylim(0, height)
    axis = pyplot.gca()
    axis.set_facecolor('blue')
    axis.axes.get_xaxis().set_visible(False)
    axis.axes.get_yaxis().set_visible(False)
    rcParams['font.size'] = 14
    rcParams['text.color'] = 'white'
    return fig, writer


def annotate_frame(i, e, average_error, example):
    pyplot.text(1, height - 1, "Iteration #" + str(i))
    pyplot.text(1, height - 2, "Training example #" + str(e + 1))
    pyplot.text(1, output_y_position, "Desired output:")
    pyplot.text(1, output_y_position - 1, str(example.output))
    pyplot.text(1, bottom_margin + 1, "Inputs:")
    pyplot.text(1, bottom_margin, str(example.inputs))
    if average_error:
        error_bar(average_error)


def error_bar(average_error):
    pyplot.text(error_bar_x_position, height - 1, "Average Error " + str(average_error) + "%")
    border = pyplot.Rectangle((error_bar_x_position, height - 3), 10, 1, color='white', fill=False)
    pyplot.gca().add_patch(border)
    rectangle = pyplot.Rectangle((error_bar_x_position, height - 3), 10 * average_error / 100, 1, color='red')
    pyplot.gca().add_patch(rectangle)

class TrainingExample():
    def __init__(self, inputs, output):
        self.inputs = inputs
        self.output = output


if __name__ == "__main__":
    #Configuration of Drawing Canvas
    vertical_distance_between_layers = 6
    horizontal_distance_between_neurons = 2
    neuron_radius = 0.5
    number_of_neurons_in_widest_layer = 4
    networkN = NeuralNetwork(number_of_neurons_in_widest_layer)
    width = 25
    height = 20
    left_margin = 10
    bottom_margin = 2
    error_bar_x_position = 14
    output_y_position = 15
    frames_per_second = 1
    metadata = dict(artist="Antonio Martinez (Metantonio)", title="Neural Network")

    #Configuration of Neural Network to draw
    print ("Configuration of Neural Network")
    #neurons_in_layers=[2,8,1]
    network = DrawNN( configurationNetwork.neurons_in_layers )
    print ("Generating an image of the neural network")
    network.draw()

    #Training Configuration section
    #training_iterations = 1000
    #show_iterations = [2, 10, 20, 50, 100, training_iterations]
    #seed_random_number_generator()
    #network = NeuralNetwork(neurons_in_layers)
    
    # Training set with inputs [a,b,c] and output
    #examples = [TrainingExample([0, 0, 1], 0),
                #TrainingExample([0, 1, 1], 1),
                #TrainingExample([1, 0, 1], 1),
                #TrainingExample([1, 1, 1], 1)]

    #Learning section
    #for i in range(1, training_iterations + 1):
            #cumulative_error = 0
            #for e, example in enumerate(examples):
                #cumulative_error += network.train(example)
                #if i in show_iterations:
                    #network.draw()
                    #annotate_frame(i, e, average_error, example)
                    #writer.grab_frame()
            #average_error = calculate_average_error(cumulative_error, len(examples))

    # Generate an image of the neural network after training
    #print ("Generating an image of the neural network after")
    #network.do_not_think()
    #network.draw()

    # Consider a new situation
    #new_situation = [1, 0, 1]
    #print ("Considering a new situation " + str(new_situation) + "?")
    #print (network.think(new_situation))
    #network.draw()
    
    os.system('python3 main2.py')
