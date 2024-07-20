import argparse
import torch
from train_module import main  # Import the main function from the new module

if __name__ == '__main__':
    # Check the number of GPUs available
    num_gpus = torch.cuda.device_count()
    
    parser = argparse.ArgumentParser(description="LLM Training Script", add_help=False)
    parser.add_argument("--distributed", action="store_true", help="Enable distributed training")
    args, unknown = parser.parse_known_args()
    
    if num_gpus > 1:
        args.distributed = True

    if args.distributed:
        print(f"Launching distributed training on {num_gpus} GPUs")
        torch.multiprocessing.spawn(main, args=(args,), nprocs=num_gpus, join=True)
    else:
        print("Launching single-GPU training")
        main(0, args)
