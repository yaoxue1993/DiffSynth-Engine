import logging
import os
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel
from torch.distributed.device_mesh import init_device_mesh

logger = logging.getLogger(__name__)


def _worker_loop(
    rank: int,
    world_size: int,
    conn,
    model: nn.Module,
    tp_plan: dict,
    device: str = "cuda",
):
    try:
        print("init process group")
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        torch.cuda.set_device(rank)
        dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
        print("init process group done")
        print(f"rank: {rank}")
        tp_mesh = init_device_mesh(device, (world_size,))
        sharded_model = parallelize_module(model, tp_mesh, tp_plan)
        print("parallelize module done")
        while True:
            data = conn.recv()
            print("recv data done")
            if data == "EXIT":
                break
            args, kwargs = data
            result = sharded_model(*args, **kwargs)
            print("forward done")
            if rank == 0:
                conn.send(result)
            dist.barrier()

    except Exception as e:
        logger.error(f"Error in worker loop (rank {rank}): {e}")
    finally:
        dist.destroy_process_group()
        conn.close()
        

class ParallelModel(torch.nn.Module):
    def __init__(
        self,
        model: nn.Module,
        tp_plan: dict,
        tp_size: int = 4,
        device: str = "cuda"
    ):
        super().__init__()
        self.world_size = tp_size
        self.device = device
        self.conn_main, self.conn_worker = mp.Pipe(duplex=True)
        mp.spawn(
            _worker_loop,
            args=(self.world_size, self.conn_worker, model, tp_plan, device),
            nprocs=self.world_size,
            join=False,
        )

    def forward(self, *args, **kwargs):
        print("start forward")
        self.conn_main.send((args, kwargs))
        return self.conn_main.recv()
            

__all__ = [
    "ParallelModel"
]