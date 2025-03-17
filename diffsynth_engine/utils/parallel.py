import logging
import os
import copy
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from typing import Optional, Union, Dict

from torch.distributed.device_mesh import init_device_mesh, DeviceMesh
from torch.distributed.tensor.parallel.style import ParallelStyle
from torch.distributed.tensor.parallel._utils import _validate_tp_mesh_dim

logger = logging.getLogger(__name__)

def wait_tensor(data):
    if isinstance(data, dict):
        return {k: wait_tensor(v) for k, v in data.items()}
    if isinstance(data, tuple) or isinstance(data, list):
        return [wait_tensor(t) for t in data]
    if hasattr(data, 'wait'):
        return data.wait()
    else:
        return data

def clone_tensor(data):
    if isinstance(data, dict):
        return {k: clone_tensor(v) for k, v in data.items()}
    if isinstance(data, tuple) or isinstance(data, list):
        return [clone_tensor(t) for t in data]
    elif isinstance(data, torch.Tensor):
        return data.clone()
    else:
        return data

def to_device(data, device):
    if isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    if isinstance(data, tuple) or isinstance(data, list):
        return [to_device(t, device) for t in data]
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data
    
def parallelize_module(        
    module: nn.Module,
    device_mesh: Optional[DeviceMesh] = None,
    parallelize_plan: Optional[Union[ParallelStyle, Dict[str, ParallelStyle]]] = None,
):
    _validate_tp_mesh_dim(device_mesh)
    if parallelize_plan is None:
        return module
    for module_path, parallelize_style in parallelize_plan.items():
        path_splits = module_path.split(".")
        if len(path_splits) == 0:
            raise ValueError("Expect module path to be non-empty, but got empty string!")
        current = module
        try:
            for atom in path_splits:
                current = getattr(current, atom)
            parallelize_style._apply(current, device_mesh)                
        except AttributeError:
            continue
    return module            

def _worker_loop(
    rank: int,
    world_size: int,
    conn,
    model: nn.Module,
    tp_plan: dict,
    master_port: int = 29500,
    device: str = "cuda",
):
    """    
    https://pytorch.org/docs/stable/multiprocessing.html#sharing-cuda-tensors
    """
    try:
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(master_port)
        torch.cuda.set_device(rank)
        dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
        tp_mesh = init_device_mesh(device, (world_size,))
        sharded_model = parallelize_module(model, tp_mesh, tp_plan)
        sharded_model.cuda()
        while True:
            if dist.get_rank() == 0:            
                args, kwargs = conn.recv()
                data = [args, kwargs]
            else:
                data = [None, None]
            dist.broadcast_object_list(data, src=0)
            args, kwargs = to_device(copy.deepcopy(data), device)
            del data
            if args == "TERMINATE" and kwargs == "TERMINATE":
                break
            with torch.no_grad():
                result = sharded_model(*args, **kwargs)
                result = wait_tensor(result)        
            if dist.get_rank() == 0:
                conn.send(result)
            dist.barrier()
    except Exception as e:
        # 打印traceback
        import traceback
        traceback.print_exc()
        logger.error(f"Error in worker loop (rank {rank}): {e}")
    finally:
        del sharded_model
        conn.close()        
        torch.cuda.synchronize()
        dist.destroy_process_group()
        

class ParallelModel(torch.nn.Module):
    def __init__(
        self,
        model: nn.Module,
        tp_plan: dict,
        tp_size: int = 4,
        master_port: int = 29500,
        device: str = "cuda"
    ):
        super().__init__()
        self.world_size = tp_size
        self.device = device
        self.conn_main, conn_worker = mp.Pipe(duplex=True)
        mp.spawn(
            _worker_loop,
            args=(self.world_size, conn_worker, model, tp_plan, master_port, device),
            nprocs=self.world_size,
            join=False,
        )

    def forward(self, *args, **kwargs):
        self.conn_main.send((args, kwargs))        
        result = self.conn_main.recv()
        return clone_tensor(result)
    
    def __del__(self):
        # Send terminate signal to all workers
        if hasattr(self, 'conn_main'):
            self.conn_main.send(("TERMINATE", "TERMINATE"))
            self.conn_main.close()          

__all__ = [
    "ParallelModel"
]