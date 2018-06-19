import multiprocessing


def map_async(f, tasks, num_processes=None):
    p = multiprocessing.Pool(processes=num_processes)
    r = p.map_async(f, tasks)
    p.close()
    p.join()
    return r.get()
