def il_mk_dataset():
    from uhtk.imitation.inputs import ILGrabber
    grabber = ILGrabber(window_keyword=None)  # full screen
    # grabber = ILGrabber(window_keyword='Phone')
    grabber.start_dataset_session()

if __name__ == '__main__':
    il_mk_dataset()