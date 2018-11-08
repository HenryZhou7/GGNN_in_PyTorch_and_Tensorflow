import os
import sys
import datetime

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
    return None

_this_dir = os.path.dirname(__file__)
_base_dir = os.path.join(_this_dir, '..')
add_path(_base_dir)

running_start_time = datetime.datetime.now()
time = str(running_start_time.strftime('%Y_%m_%d-%X'))


def get_base_dir():
    return _base_dir

def get_time():
    return time