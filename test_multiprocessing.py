import multiprocessing as mp
import time
import math

results_a = []
results_b = []
results_c = []

def make_calculation_one(numbers):
    for number in numbers:
        results_a.append(math.sqrt(number**3))

def make_calculation_two(numbers):
    for number in numbers:
        results_b.append(math.sqrt(number**4))

def make_calculation_three(numbers):
    for number in numbers:
        results_c.append(math.sqrt(number**5))