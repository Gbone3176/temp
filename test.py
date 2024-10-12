
import os
import numpy as np

def print_name():
    print(__name__)
    print(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if __name__ == '__main__':
    a = np.array([1,2,3])
    b = "tests/models/test_output/01.mp4"

    print(os.path.split(b)[0])


