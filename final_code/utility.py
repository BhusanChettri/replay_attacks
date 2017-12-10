'''
This file contains different utility helper functions..
'''

import sys
import os

def create_directory(directory):
    
    # Create directories for storing tensorflow events
    if os.path.exists(directory):
        shutil.rmtree(directory)
    else:
        os.makedirs(directory)        
    return 0


def makeDirectory(path):
    # Create directories for storing tensorflow events    
    if not os.path.exists(path):
        os.makedirs(path)