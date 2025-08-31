import pprint
import numpy as np

def print_obj(obj):
    for attr in dir(obj):
        if not attr.startswith('__'):
            value = getattr(obj, attr)
            print(f"{attr}: {value}")
            # print(f"{attr}")


def print_dict(data):
    summary = {
        key: f" {type(value)}, shape={value.shape}, dtype={value.dtype}" if isinstance(value, np.ndarray) 
                                                                         else type(value) 
                                                                         for key, value in data.items()
    }
    pprint.pp(summary)


def print_list(data):
    assert isinstance(data, list)
    print("list len: ", len(data))
    # item_len = []
    # for item in data: item_len.append(len(item))
    # print(item_len)
    print("[", end="")
    for index, item in enumerate(data): 
        print(f" {index}: ", end="")
        print_dict(vars(item))
    print("]")