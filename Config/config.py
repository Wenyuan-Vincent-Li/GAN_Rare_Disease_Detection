import math
import numpy as np

# Base Configuration Class
# Don't use this class directly. Instead, sub-class it and override
# the configurations you need to change.

class Config(object):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """
    # Name the configurations. For example, 'GAN', 'Experiment 3', ...etc.
    # Useful if your code needs to do things differently depending on which
    # experiment is running.
    NAME = None  # Override in sub-classes
    
    ## Input pipeline
    DATA_DIR = None # Specify the data directory, override in sub-classes
    FILE_MID_NAME = None # Specify the data directory, override in sub-classes
    BATCH_SIZE = 512 # Batch size
    BATCH_RATIO = 3 # Negtive samples are 3 times the size of positive samples
    
    ## Model architecture
    # The hidden layer number of neurons for the discriminator
    DISCRIMINATOR_NUM_HL = [1000, 500, 250, 250, 250]
    # The hidden layer number of neurons for the generator
    # The first number is the length of the latent variable z
    GENERATOR_NUM_HL = [100, 500, 500, 500]
    # Feature lenght of the data
    FEATURE_NUM = 797
    # Number of classification classes
    NUM_CLASSES = 2  # [positive, negative]
    
    ## Training settings
    # Restore
    RESTORE = False # Whether to use the previous trained weights
    PRE_TRAIN = False # Whether to use the pre-trained weights for discriminator
    PRE_TRAIN_FILE_PATH = None # Specify the pre-trained weights file path
    # Optimizer
    BATCH_NORM_DECAY =  0.999
    BATCH_NORM_EPSILON = 0.001
    LEARNING_RATE = 1e-3
    MOMENTUM = 0.5
    # Loss settings
    WEIGHT_REG = 1e-4 # L2 Weight regularization constant
    PULL_AWAY = False # Whether to use pull away term
    ENTROPY = False # Whether to use the entropy term
    # Training schedule
    EPOCHS = None # Num of epochs to train in the current run, override in sub-classes
    TRAIN_SIZE = None # Num of samples used to train per epoch, override in sub-classes
    VAL_SIZE = None # Num of samples used to val after every epoch, override in sub-classes
    SAVE_PER_EPOCH = 2 # How often to save the trained weights
    # Summary
    SUMMARY = True
    SUMMARY_GRAPH = True
    SUMMARY_SCALAR = True
    SUMMARY_IMAGE = False
    SUMMARY_TRAIN_VAL = True
    SUMMARY_HISTOGRAM = False
    
    def __init__(self):
      """Set values of computed attributes."""
      self.MIN_QUEUE_EXAMPLES = self.BATCH_SIZE * 3

    def display(self):
      """Display Configuration values."""
      print("\nConfigurations:")
      for a in dir(self):
        if not a.startswith("__") and not callable(getattr(self, a)):
          print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
    
    def config_str(self):
      """Return a configurations string"""
      s = "\nConfigurations:\n"
      for a in dir(self):
        if not a.startswith("__") and not callable(getattr(self, a)):
          s += "{:30} {}".format(a, getattr(self, a))
          s += "\n"
      return s
          
if __name__ == "__main__":
  tmp_config = Config()
  s = tmp_config.config_str()