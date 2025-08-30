import psutil
from uhtk.UTIL.colorful import *

def kill_process(p):
    try:
        # print('正在发送terminate命令到进程:', os.getpid(), '-->', p.pid)
        p.terminate()
        _, alive = psutil.wait_procs([p,], timeout=0.01)    # 先等 10ms
        if len(alive):
            _, alive = psutil.wait_procs(alive, timeout=0.10)  # 再等 100ms
            if len(alive):
                # print('\t (R1) 很遗憾, 进程不服从terminate信号, 正在发送kill-9命令到进程:', os.getpid(), '-->', p.pid)
                for p in alive: p.kill()
            else:
                # print('\t (R2) 进程成功结束')
                pass
        else:
            # print('\t (R2) 进程成功结束')
            pass
    except Exception as e:
        print(e) 

def kill_process_and_its_children(p):
    p = psutil.Process(p.pid)   # p might be Python's process, convert to psutil's process
    if len(p.children())>0:
        # print('有子进程')
        for child in p.children():
            if hasattr(child,'children') and len(child.children())>0:
                kill_process_and_its_children(child)
            else:
                kill_process(child)
    else:
        pass
        # print('无子进程')
    kill_process(p)


def kill_process_children(p):
    p = psutil.Process(p.pid)   # p might be Python's process, convert to psutil's process
    if len(p.children())>0:
        # print('有子进程')
        for child in p.children():
            if hasattr(child,'children') and len(child.children())>0:
                kill_process_and_its_children(child)
            else:
                kill_process(child)
    else:
        pass
        # print('无子进程')

def clean_child_process(pid):
    parent = psutil.Process(pid)
    kill_process_children(parent)