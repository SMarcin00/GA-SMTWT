import math
import numpy as np


class RandomNumberGenerator:
    def __init__(self, seedVaule=None):
        self.__seed = seedVaule

    def nextInt(self, low, high):
        m = 2147483647
        a = 16807
        b = 127773
        c = 2836
        k = int(self.__seed / b)
        self.__seed = a * (self.__seed % b) - k * c
        if self.__seed < 0:
            self.__seed = self.__seed + m
        value_0_1 = self.__seed
        value_0_1 = value_0_1 / m
        return low + int(math.floor(value_0_1 * (high - low + 1)))

    def nextFloat(self, low, high):
        low *= 100000
        high *= 100000
        val = self.nextInt(low, high) / 100000.0
        return val

task_number = 3
seed =123

def generate_smtwt_instance(task_number, seed=123):
    """Function to generate smtwt instance

    Args:
        task_number (_type_): number of task given to the machine
        seed (int, optional): generator seed

    Returns:
        p,w,d: execution time, weight, deadline
    """
    p = np.zeros(task_number)
    w = np.zeros(task_number)
    d = np.zeros(task_number)
    
    g = RandomNumberGenerator(seedVaule=seed)
    
    for i in range(task_number):
        p[i] = g.nextInt(1,30)
        w[i] = g.nextInt(1,30)
    
    S = np.sum(p)
    
    for i in range(task_number):
        d[i] = g.nextInt(1,S)
        
    return p, w, d

