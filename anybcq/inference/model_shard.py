import json
import os
from typing import Any, Optional, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import ProcessGroup

_DTYPE_TO_CODE = {
    torch.float16: 0,
    torch.bfloat16: 1,
    torch.float32: 2,
    torch.float64: 3,
    torch.int64: 4,
    torch.int32: 5,
    torch.int16: 6,
    torch.int8: 7,
    torch.uint8: 8,
    torch.bool: 9,
}
_CODE_TO_DTYPE = {code: dtype for dtype, code in _DTYPE_TO_CODE.items()}


def _require_process_group(process_group: Optional[ProcessGroup]) -> ProcessGroup:
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed is not initialized.")
    return process_group if process_group is not None else dist.group.WORLD


def _default_comm_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _infer_module_device(module: nn.Module) -> torch.device:
    for tensor in list(module.parameters()) + list(module.buffers()):
        return tensor.device
    return _default_comm_device()


def _validate_tensor_backend_device(tensor: torch.Tensor, process_group: ProcessGroup):
    backend = dist.get_backend(process_group)
    if backend == "nccl" and tensor.device.type != "cuda":
        raise ValueError("NCCL backend requires CUDA tensors for send/recv payloads.")


def _send_json(payload: dict[str, Any], dst: int, device: torch.device, process_group: ProcessGroup):
    payload_bytes = json.dumps(payload).encode("utf-8")
    byte_tensor = torch.tensor(list(payload_bytes), dtype=torch.uint8, device=device)
    size = torch.tensor([byte_tensor.numel()], dtype=torch.int64, device=device)
    _validate_tensor_backend_device(size, process_group)
    dist.send(size, dst=dst, group=process_group)
    dist.send(byte_tensor, dst=dst, group=process_group)


def _recv_json(src: int, device: torch.device, process_group: ProcessGroup):
    size = torch.empty((1,), dtype=torch.int64, device=device)
    _validate_tensor_backend_device(size, process_group)
    dist.recv(size, src=src, group=process_group)
    numel = int(size.item())
    payload = torch.empty((numel,), dtype=torch.uint8, device=device)
    dist.recv(payload, src=src, group=process_group)
    decoded = bytes(payload.cpu().tolist()).decode("utf-8")
    return json.loads(decoded)


def _send_tensor(tensor: torch.Tensor, dst: int, process_group: ProcessGroup):
    if tensor.dtype not in _DTYPE_TO_CODE:
        raise ValueError(f"Unsupported tensor dtype for transport: {tensor.dtype}")
    tensor = tensor.contiguous()
    _validate_tensor_backend_device(tensor, process_group)

    metadata = torch.tensor(
        [tensor.dim(), _DTYPE_TO_CODE[tensor.dtype]], dtype=torch.int64, device=tensor.device
    )
    dist.send(metadata, dst=dst, group=process_group)
    if tensor.dim() > 0:
        shape = torch.tensor(tensor.shape, dtype=torch.int64, device=tensor.device)
        dist.send(shape, dst=dst, group=process_group)
    dist.send(tensor, dst=dst, group=process_group)


def _recv_tensor(src: int, device: torch.device, process_group: ProcessGroup):
    metadata = torch.empty((2,), dtype=torch.int64, device=device)
    _validate_tensor_backend_device(metadata, process_group)
    dist.recv(metadata, src=src, group=process_group)
    ndim = int(metadata[0].item())
    dtype_code = int(metadata[1].item())
    dtype = _CODE_TO_DTYPE.get(dtype_code)
    if dtype is None:
        raise ValueError(f"Unsupported received dtype code: {dtype_code}")

    if ndim > 0:
        shape = torch.empty((ndim,), dtype=torch.int64, device=device)
        dist.recv(shape, src=src, group=process_group)
        recv_shape = tuple(int(v) for v in shape.tolist())
    else:
        recv_shape = ()

    tensor = torch.empty(recv_shape, dtype=dtype, device=device)
    dist.recv(tensor, src=src, group=process_group)
    return tensor


def _tensor_from_layer_output(layer_output: Any) -> torch.Tensor:
    if isinstance(layer_output, torch.Tensor):
        return layer_output
    if isinstance(layer_output, (tuple, list)):
        for value in layer_output:
            if isinstance(value, torch.Tensor):
                return value
    if hasattr(layer_output, "last_hidden_state") and isinstance(
        layer_output.last_hidden_state, torch.Tensor
    ):
        return layer_output.last_hidden_state
    raise TypeError(
        "Shard module output must contain a tensor hidden state "
        "(Tensor, tuple/list with tensor, or object with last_hidden_state)."
    )


class ModelShard(nn.Module):
    def __init__(
        self,
        module: Optional[nn.Module] = None,
        shard_id: int = 0,
        src_rank: Optional[int] = None,
        dst_rank: Optional[int] = None,
        process_group: Optional[ProcessGroup] = None,
        auto_init_process_group: bool = False,
        dist_backend: str = "nccl",
        dist_init_method: str = "env://",
    ):
        super().__init__()
        self.module = module if module is not None else nn.Identity()
        self.shard_id = shard_id
        self.src_rank = src_rank
        self.dst_rank = dst_rank
        self.process_group = process_group
        if self.process_group is None and dist.is_initialized():
            self.process_group = dist.group.WORLD
        if auto_init_process_group and self.process_group is None:
            self.initialize_process_group_from_env(
                backend=dist_backend, init_method=dist_init_method
            )

    def initialize_process_group_from_env(
        self, backend: str = "nccl", init_method: str = "env://"
    ) -> ProcessGroup:
        if dist.is_initialized():
            self.process_group = dist.group.WORLD
            return self.process_group

        world_size_str = os.getenv("WORLD_SIZE")
        rank_str = os.getenv("RANK")
        if world_size_str is None or rank_str is None:
            raise ValueError("WORLD_SIZE and RANK must be set before distributed initialization.")

        world_size = int(world_size_str)
        rank = int(rank_str)
        local_rank_str = os.getenv("LOCAL_RANK")
        if local_rank_str is not None and torch.cuda.is_available():
            torch.cuda.set_device(int(local_rank_str))

        dist.init_process_group(
            backend=backend, init_method=init_method, world_size=world_size, rank=rank
        )
        self.process_group = dist.group.WORLD
        return self.process_group

    def forward(self, hidden_states: torch.Tensor, **forward_kwargs) -> torch.Tensor:
        layer_output = self.module(hidden_states, **forward_kwargs)
        return _tensor_from_layer_output(layer_output)

    def send_packet(
        self,
        hidden_states: torch.Tensor,
        side_tensors: Optional[dict[str, torch.Tensor]] = None,
        dst_rank: Optional[int] = None,
    ):
        process_group = _require_process_group(self.process_group)
        target_rank = self.dst_rank if dst_rank is None else dst_rank
        if target_rank is None:
            raise ValueError("Destination rank is not set for this model shard.")

        side_tensors = side_tensors or {}
        for key, value in side_tensors.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"side_tensors['{key}'] must be a torch.Tensor.")

        _send_json(
            {"done": False, "side_keys": list(side_tensors.keys())},
            dst=target_rank,
            device=hidden_states.device,
            process_group=process_group,
        )
        _send_tensor(hidden_states, dst=target_rank, process_group=process_group)
        for key in side_tensors:
            _send_tensor(side_tensors[key], dst=target_rank, process_group=process_group)

    def send_done(self, dst_rank: Optional[int] = None):
        process_group = _require_process_group(self.process_group)
        target_rank = self.dst_rank if dst_rank is None else dst_rank
        if target_rank is None:
            raise ValueError("Destination rank is not set for this model shard.")
        comm_device = _infer_module_device(self.module)
        _send_json(
            {"done": True},
            dst=target_rank,
            device=comm_device,
            process_group=process_group,
        )

    def recv_packet(
        self, src_rank: Optional[int] = None, device: Optional[torch.device] = None
    ) -> tuple[bool, Optional[torch.Tensor], dict[str, torch.Tensor]]:
        process_group = _require_process_group(self.process_group)
        source_rank = self.src_rank if src_rank is None else src_rank
        if source_rank is None:
            raise ValueError("Source rank is not set for this model shard.")
        recv_device = _infer_module_device(self.module) if device is None else device

        metadata = _recv_json(src=source_rank, device=recv_device, process_group=process_group)
        if bool(metadata.get("done", False)):
            return True, None, {}

        hidden_states = _recv_tensor(src=source_rank, device=recv_device, process_group=process_group)
        side_tensors: dict[str, torch.Tensor] = {}
        for key in metadata.get("side_keys", []):
            side_tensors[key] = _recv_tensor(
                src=source_rank, device=recv_device, process_group=process_group
            )
        return False, hidden_states, side_tensors

    def pass_tensor_to_rank(self, node: int, x: torch.Tensor):
        self.send_packet(hidden_states=x, dst_rank=node)

    def receive_tensor_from_rank(
        self, node: int, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        done, hidden_states, _ = self.recv_packet(src_rank=node, device=device)
        if done or hidden_states is None:
            raise RuntimeError(f"Received done packet while waiting for tensor from rank {node}.")
        return hidden_states

    def run_stage(
        self,
        hidden_states: Optional[torch.Tensor] = None,
        side_tensors: Optional[dict[str, torch.Tensor]] = None,
        forward_kwargs: Optional[dict[str, Any]] = None,
        pass_side_tensors_to_forward: bool = False,
        src_rank: Optional[int] = None,
        dst_rank: Optional[int] = None,
        recv_device: Optional[torch.device] = None,
    ) -> Optional[tuple[torch.Tensor, dict[str, torch.Tensor]]]:
        packet_side = dict(side_tensors or {})
        current_hidden_states = hidden_states

        if current_hidden_states is None:
            done, current_hidden_states, packet_side = self.recv_packet(
                src_rank=src_rank, device=recv_device
            )
            if done or current_hidden_states is None:
                target_rank = self.dst_rank if dst_rank is None else dst_rank
                if target_rank is not None:
                    self.send_done(dst_rank=target_rank)
                return None

        layer_kwargs = dict(forward_kwargs or {})
        if pass_side_tensors_to_forward:
            for key, value in packet_side.items():
                layer_kwargs.setdefault(key, value)

        current_hidden_states = self(current_hidden_states, **layer_kwargs)
        target_rank = self.dst_rank if dst_rank is None else dst_rank
        if target_rank is not None:
            self.send_packet(
                hidden_states=current_hidden_states,
                side_tensors=packet_side,
                dst_rank=target_rank,
            )
            return None
        return current_hidden_states, packet_side

    def serve(
        self,
        *,
        src_rank: Optional[int] = None,
        dst_rank: Optional[int] = None,
        forward_kwargs: Optional[dict[str, Any]] = None,
        pass_side_tensors_to_forward: bool = False,
        recv_device: Optional[torch.device] = None,
    ) -> Optional[list[tuple[torch.Tensor, dict[str, torch.Tensor]]]]:
        target_rank = self.dst_rank if dst_rank is None else dst_rank
        collected_outputs: list[tuple[torch.Tensor, dict[str, torch.Tensor]]] = []
        while True:
            done, hidden_states, packet_side = self.recv_packet(
                src_rank=src_rank, device=recv_device
            )
            if done or hidden_states is None:
                if target_rank is not None:
                    self.send_done(dst_rank=target_rank)
                    return None
                return collected_outputs

            layer_kwargs = dict(forward_kwargs or {})
            if pass_side_tensors_to_forward:
                for key, value in packet_side.items():
                    layer_kwargs.setdefault(key, value)
            hidden_states = self(hidden_states, **layer_kwargs)

            if target_rank is not None:
                self.send_packet(
                    hidden_states=hidden_states,
                    side_tensors=packet_side,
                    dst_rank=target_rank,
                )
            else:
                collected_outputs.append((hidden_states, packet_side))


class ModelShardPipeline(nn.Module):
    def __init__(self, shards: Sequence[ModelShard]):
        if len(shards) == 0:
            raise ValueError("ModelShardPipeline requires at least one shard.")
        super().__init__()
        self.shards = nn.ModuleList(shards)

    @classmethod
    def from_modules(
        cls,
        modules: Sequence[nn.Module],
        *,
        start_rank: int = 0,
        process_group: Optional[ProcessGroup] = None,
        auto_init_process_group: bool = False,
        dist_backend: str = "nccl",
        dist_init_method: str = "env://",
    ):
        if len(modules) == 0:
            raise ValueError("from_modules requires at least one module.")
        shards = []
        for idx, module in enumerate(modules):
            src_rank = None if idx == 0 else start_rank + idx - 1
            dst_rank = None if idx == (len(modules) - 1) else start_rank + idx + 1
            shards.append(
                ModelShard(
                    module=module,
                    shard_id=idx,
                    src_rank=src_rank,
                    dst_rank=dst_rank,
                    process_group=process_group,
                    auto_init_process_group=auto_init_process_group,
                    dist_backend=dist_backend,
                    dist_init_method=dist_init_method,
                )
            )
        return cls(shards)

    def forward(
        self,
        hidden_states: torch.Tensor,
        shard_kwargs: Optional[Sequence[dict[str, Any]]] = None,
        **forward_kwargs,
    ) -> torch.Tensor:
        if shard_kwargs is not None and len(shard_kwargs) != len(self.shards):
            raise ValueError("shard_kwargs must match the number of shards.")

        out = hidden_states
        for idx, shard in enumerate(self.shards):
            call_kwargs = dict(forward_kwargs)
            if shard_kwargs is not None and shard_kwargs[idx]:
                call_kwargs.update(shard_kwargs[idx])
            out = shard(out, **call_kwargs)
        return out

    def run_distributed_stage(
        self,
        hidden_states: Optional[torch.Tensor] = None,
        side_tensors: Optional[dict[str, torch.Tensor]] = None,
        shard_kwargs: Optional[Sequence[dict[str, Any]]] = None,
        pass_side_tensors_to_forward: bool = False,
        recv_device: Optional[torch.device] = None,
        **forward_kwargs,
    ) -> Optional[tuple[torch.Tensor, dict[str, torch.Tensor]]]:
        if shard_kwargs is not None and len(shard_kwargs) != len(self.shards):
            raise ValueError("shard_kwargs must match the number of shards.")

        packet_side = dict(side_tensors or {})
        out = hidden_states
        if out is None:
            done, out, packet_side = self.shards[0].recv_packet(device=recv_device)
            if done or out is None:
                last = self.shards[-1]
                if last.dst_rank is not None:
                    last.send_done()
                return None

        for idx, shard in enumerate(self.shards):
            call_kwargs = dict(forward_kwargs)
            if shard_kwargs is not None and shard_kwargs[idx]:
                call_kwargs.update(shard_kwargs[idx])
            if pass_side_tensors_to_forward:
                for key, value in packet_side.items():
                    call_kwargs.setdefault(key, value)
            out = shard(out, **call_kwargs)

        last = self.shards[-1]
        if last.dst_rank is not None:
            last.send_packet(out, side_tensors=packet_side)
            return None
        return out, packet_side

    def serve_stage(
        self,
        *,
        shard_kwargs: Optional[Sequence[dict[str, Any]]] = None,
        pass_side_tensors_to_forward: bool = False,
        recv_device: Optional[torch.device] = None,
        **forward_kwargs,
    ) -> Optional[list[tuple[torch.Tensor, dict[str, torch.Tensor]]]]:
        if shard_kwargs is not None and len(shard_kwargs) != len(self.shards):
            raise ValueError("shard_kwargs must match the number of shards.")

        first = self.shards[0]
        last = self.shards[-1]
        target_rank = last.dst_rank
        collected_outputs: list[tuple[torch.Tensor, dict[str, torch.Tensor]]] = []

        while True:
            done, hidden_states, packet_side = first.recv_packet(device=recv_device)
            if done or hidden_states is None:
                if target_rank is not None:
                    last.send_done(dst_rank=target_rank)
                    return None
                return collected_outputs

            out = hidden_states
            for idx, shard in enumerate(self.shards):
                call_kwargs = dict(forward_kwargs)
                if shard_kwargs is not None and shard_kwargs[idx]:
                    call_kwargs.update(shard_kwargs[idx])
                if pass_side_tensors_to_forward:
                    for key, value in packet_side.items():
                        call_kwargs.setdefault(key, value)
                out = shard(out, **call_kwargs)

            if target_rank is not None:
                last.send_packet(out, side_tensors=packet_side, dst_rank=target_rank)
            else:
                collected_outputs.append((out, packet_side))


class Model_Shard(ModelShard):
    def __init__(
        self,
        block_size: int = 2048,
        vocab_size: int = 32000,
        n_layer: int = 32,
        n_head: int = 32,
        dim: int = 4096,
        intermediate_size: Optional[int] = None,
        n_local_heads: int = -1,
        head_dim: int = 64,
        rope_base: float = 10000,
        norm_eps: float = 1e-5,
        rope_scaling: Optional[dict] = None,
        model_name: Optional[str] = None,
        module: Optional[nn.Module] = None,
        shard_id: int = 0,
        src_rank: Optional[int] = None,
        dst_rank: Optional[int] = None,
        process_group: Optional[ProcessGroup] = None,
        auto_init_process_group: bool = True,
        dist_backend: str = "nccl",
        dist_init_method: str = "env://",
    ):
        super().__init__(
            module=module,
            shard_id=shard_id,
            src_rank=src_rank,
            dst_rank=dst_rank,
            process_group=process_group,
            auto_init_process_group=auto_init_process_group,
            dist_backend=dist_backend,
            dist_init_method=dist_init_method,
        )
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.dim = dim
        self.intermediate_size = intermediate_size
        self.n_local_heads = n_local_heads
        self.head_dim = head_dim
        self.rope_base = rope_base
        self.norm_eps = norm_eps
        self.rope_scaling = rope_scaling
        self.model_name = model_name
        
