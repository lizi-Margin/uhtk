import os, threading

def get_root_dir():
    # root_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.getcwd() + '/'
    return root_dir

class GrabberConfig:
    root_dir = get_root_dir()
    tick = 0.1

    # SIRI Standard
    # sz_wh = (1280, 578)

    # 2560 x 1600
    # sz_wh = (320, 200)
    # sz_wh = (960, 600)

    # 2560 x 1440
    # sz_wh = (320, 180)
    sz_wh = (960, 540)

class GrabberStatus:
    monitor = None
    stop_event = threading.Event()

    @classmethod
    def in_window_center_xy(cls):
        assert cls.monitor is not None
        return (cls.monitor['width']/2, cls.monitor['height']/2)