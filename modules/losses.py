import os
import matplotlib.pyplot as plt

class LossCounter():
    def __init__(self):
        self.losses = {'train':[], 'val':[]}

    def add(self, phase, loss):
        self.losses[phase].append(loss)

    def plot_loss(self, result_dir):
        # Plot the loss values.
        plt.plot(self.losses['train'], label='Train')
        plt.plot(self.losses['val'], label='Val')

        # Set the title and axis labels.
        plt.title('Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Show the plot.
        plt.savefig(os.path.join(result_dir, "loss.png"))