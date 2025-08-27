def load_dataset(traj_dir='20250826-23#26#53'):
    from uhtk.imitation.utils import safe_load_traj_pool
    from uhtk.siri.utils.print_anything import print_list

    load = safe_load_traj_pool(traj_dir=traj_dir)
    x = load()
    print_list(x)