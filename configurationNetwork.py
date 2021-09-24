#layers can be [input, hidden, hidden,...,hidden,...,output]
neurons_in_layers=[3,8,8,1]

class TrainingExample():
    def __init__(self, inputs, output):
        self.inputs = inputs
        self.output = output

#Example of training data
examples = [TrainingExample([0, 0, 1], 0),
                TrainingExample([0, 1, 1], 1),
                TrainingExample([1, 0, 1], 1),
                TrainingExample([1, 1, 1], 1)]

training_iterations = 1000
show_iterations = [2, 10, 20, 50, 100, training_iterations]

#New situation to evaluate in neuronal network
new_situation = [1, 0, 1]

video_file_name = "neural_network.mp4"
