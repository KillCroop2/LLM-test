import argparse
import torch
import os
from train_module import main  # Import the main function from the new module

if __name__ == '__main__':
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        # Set environment variables for distributed training
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        # Spawn processes
        torch.multiprocessing.spawn(main, args=(num_gpus,), nprocs=num_gpus)
    else:
        main()