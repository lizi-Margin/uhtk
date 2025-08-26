import os, threading

def get_root_dir():
    # root_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.getcwd() + '/'
    return root_dir

class GrabberConfig:
    root_dir = get_root_dir()
    tick = 0.1
    # sz_wh = (1280, 578)
    sz_wh = (320, 200)

class GrabberStatus:
    monitor = None
    stop_event = threading.Event()

    @classmethod
    def in_window_center_xy(cls):
        assert cls.monitor is not None
        return (cls.monitor['width']/2, cls.monitor['height']/2)