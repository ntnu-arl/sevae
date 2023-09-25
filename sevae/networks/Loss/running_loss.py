import math

class RunningLoss:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.aggregated_loss = 0.0
        self.num_samples = 0
        self.reset()
    
    def update(self, loss):
        self.aggregated_loss += loss/self.batch_size
        self.num_samples += 1
        self.avg = self.aggregated_loss / (self.num_samples)
    
    def reset(self):
        self.aggregated_loss = 0.0
        self.num_samples = 0
        self.avg = 0.0
    
