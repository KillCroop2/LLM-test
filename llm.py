import torch
import os
from train_module import main

if __name__ == '__main__':
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        torch.multiprocessing.spawn(main, args=(num_gpus,), nprocs=num_gpus)
    else:
        main(local_rank=0, world_size=1)