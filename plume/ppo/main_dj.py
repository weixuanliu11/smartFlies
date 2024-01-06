from plume.schema import *
from multiprocessing import Process, Pipe
import sys

if __name__ == '__main__':
    num_of_processes = sys.argv[1]
    processes = []
    kw={'reserve_jobs':True}
    for i in num_of_processes:
        processes.append(Process(target=TrainingResult.populate, kwargs=kw))

    for p in processes:
        p.start()

    for p in processes:
        p.join()