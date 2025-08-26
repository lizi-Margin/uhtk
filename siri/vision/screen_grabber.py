import os
import cv2
import mss
import time
import platform
import subprocess
import numpy as np

from uhtk.siri.grabber_config import GrabberConfig as cfg
from uhtk.siri.grabber_config import GrabberStatus
from uhtk.siri.utils.lprint import lprint
from uhtk.siri.utils.sleeper import Sleeper

class ScrGrabber():
    def __init__(self, window_keyword=None):
        self.window_keyword = window_keyword

    def get_scrcpy_window_geometry(self):
        if self.window_keyword is None:  # full scrren
            from screeninfo import get_monitors
            monitor = get_monitors()[0]
            return 0, 0, monitor.width, monitor.height

        # if platform is linux
        if platform.system()=="Linux":
            result = subprocess.run(
                ['wmctrl', '-lG'],
                stdout=subprocess.PIPE,
                text=True
            )
            lines = result.stdout.splitlines()
            for line in lines:
                if self.window_keyword in line:
                    parts = line.split()
                    x, y = int(parts[2]), int(parts[3])
                    scr_width, scr_height = int(parts[4]), int(parts[5])
                    return x, y, scr_width, scr_height
            return None
        elif platform.system()=="Windows":
            return None
        else:
            return None

    @staticmethod
    def sync_monitor(geometry):
        left, top, scr_width, scr_height = geometry
        GrabberStatus.monitor = {
            "top": top,
            "left": left,
            "width": scr_width,
            "height": scr_height,
        }
        lprint("ScrGrabber", f"sync_monitor: {GrabberStatus.monitor}")
    
    def sync_monitor_every_n_step(self, n=100):
        if hasattr(self, "sync_monitor_counter"):
            self.sync_monitor_counter += 1
        else:
            self.sync_monitor_counter = 0
        if self.sync_monitor_counter % n == 0:
            geometry = self.get_scrcpy_window_geometry()
            if not geometry:
                lprint(self, "scrcpy window not found")
            else:
                self.sync_monitor(geometry)
            self.sync_monitor_counter = 0

    def start_session(self, func, *args, **kwargs):
        geometry = self.get_scrcpy_window_geometry()
        if not geometry:
            lprint(self, "scrcpy window not found")
            lprint(self, "start_session failed")
            return

        assert GrabberStatus.monitor is None
        self.sync_monitor(geometry)
        

        try:
            with mss.mss() as sct:
                while True:
                    sleeper = Sleeper(tick=cfg.tick, user=self)
                    screenshot = sct.grab(GrabberStatus.monitor)
                    frame = np.array(screenshot)
                    if frame.shape[-1] == 4:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    assert frame.shape[-1] == 3, 'not a BGR format'
                    frame = cv2.resize(frame, cfg.sz_wh)

                    func(frame, *args, **kwargs)

                    self.sync_monitor_every_n_step(n=100)
                    sleeper.sleep()
        except KeyboardInterrupt:
            lprint(self, "Sig INT catched, stopping session.")
        finally:
            # cv2.destroyAllWindows()
            GrabberStatus.monitor = None
            GrabberStatus.stop_event.set()
    
    def save_frame(self, frame: np.ndarray):
        assert isinstance(frame, np.ndarray)

        save_dir = f"{cfg.root_dir}/{self.__class__.__name__}"
        os.makedirs(save_dir, exist_ok=True)

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(save_dir, f"frame_{timestamp}.png")
        cv2.imwrite(filename, frame)
    
    def start_capture_session(self):
        self.start_session(func=self.save_frame)

