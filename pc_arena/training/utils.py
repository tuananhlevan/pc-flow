import torch
import os
import socket
import shutil
import re

def ddp_setup(rank: int, world_size: int, port: int):
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = str(port)

    backend = 'gloo' if os.name == 'nt' else 'nccl'
    torch.distributed.init_process_group(backend = backend, rank = rank, world_size = world_size)


def get_free_port():
    sock = socket.socket()
    sock.bind(('', 0))
    return sock.getsockname()[1]


def mkdir_p(path):
    os.makedirs(path, exist_ok=True)


def copy_configs(path, args):
    mkdir_p(path)
    shutil.copy(os.path.join("./configs/data/", args.data_config + ".yaml"), os.path.join(path, "data_config.yaml"))
    shutil.copy(os.path.join("./configs/model/", args.model_config + ".yaml"), os.path.join(path, "model_config.yaml"))
    shutil.copy(os.path.join("./configs/optim/", args.optim_config + ".yaml"), os.path.join(path, "optim_config.yaml"))


def find_largest_epoch(file_path):
    # Regular expression to match epoch numbers
    epoch_pattern = re.compile(r'\[Epoch (\d+)\]')
    
    # Read the file content
    with open(file_path, 'r') as file:
        file_content = file.read()
    
    # Find all epoch numbers
    epochs = epoch_pattern.findall(file_content)
    
    # Convert epoch numbers to integers and find the maximum
    if epochs:
        epochs = list(map(int, epochs))
        largest_epoch = max(epochs)
        return largest_epoch
    else:
        return None


def resolve_tuple(*args):
    return tuple(args)