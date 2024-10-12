from test import print_name
import os
print_name()

print(__name__)
print(__file__)
print(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
print(os.path.join(os.path.dirname(__file__), ".."))
print(os.path.dirname(__file__), "..")

'''
output:

test
/home/PJLAB/guobowen/workspace
__main__
/home/PJLAB/guobowen/workspace/sketchs/test1.py
/home/PJLAB/guobowen/workspace
/home/PJLAB/guobowen/workspace/sketchs/..
/home/PJLAB/guobowen/workspace/sketchs ..
'''