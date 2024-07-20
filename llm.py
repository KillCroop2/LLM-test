import argparse
import torch
from train_module import main  # Import the main function from the new module

if __name__ == '__main__':
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        torch.multiprocessing.spawn(main, nprocs=num_gpus)
    else:
        main()
