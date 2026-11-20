# SPDX-License-Identifier: Apache-2.0

from typing import Optional
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.distributed.device_communicators.base_device_communicator \
    import DeviceCommunicatorBase
from vllm.distributed.parallel_state import GroupCoordinator, get_dp_group, get_tp_group, get_ep_group

import habana_frameworks.torch as htorch  # noqa: F401

from vllm_gaudi.v1.worker.hpu_dp_utils import get_hpu_dp_metadata


class HpuCommunicator(DeviceCommunicatorBase):

    def __init__(self,
                 cpu_group: ProcessGroup,
                 device: Optional[torch.device] = None,
                 device_group: Optional[ProcessGroup] = None,
                 unique_name: str = "",
                 **kwargs):
        # Forward any extra keyword arguments (e.g. ``global_ranks``,
        # ``global_world_size``, ``use_all2all``) to the upstream base
        # constructor. Upstream repeatedly churns this signature, so we
        # capture the trailing params via ``**kwargs`` rather than pinning a
        # fixed list that breaks whenever vLLM adds or removes one.
        super().__init__(cpu_group, device, device_group, unique_name, **kwargs)

        self.dp_group: Optional[GroupCoordinator] = None
        self.dp_rank = 0
        self.dp_world_size = 1
        # assume EP is enabled along with DP
        if "ep" in unique_name:
            self.dp_group = get_dp_group()
            self.dp_rank = self.dp_group.rank_in_group
            self.dp_world_size = self.dp_group.world_size
            self.tp_group = get_tp_group()
        self.world_size = dist.get_world_size(group=self.cpu_group)
        self.rank = dist.get_rank(group=self.cpu_group)

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        from vllm_gaudi import envs as gaudi_envs
        if (gaudi_envs.VLLM_HPU_DSV4_NATIVE_DECODE_GRAPH and torch.compiler.is_compiling()
                and input_.shape == (1, 4096) and input_.dtype == torch.bfloat16):
            return torch.ops.vllm_gaudi.tp2_allreduce_plain(input_)
        gather_rows = int(gaudi_envs.VLLM_HPU_ALL_REDUCE_AS_GATHER_ROWS)
        if (gather_rows > 0 and input_.dim() >= 1 and input_.is_floating_point() and input_.shape[-1] > 0
                and 0 < input_.numel() // input_.shape[-1] <= gather_rows):  # rows: every dim but the last
            return self._all_reduce_as_gather(input_)
        # FIXME(kzawora): this is a workaround for a bug in Habana PT bridge
        # occurring when PT_HPU_ENABLE_LAZY_COLLECTIVES=true env var is used
        # (which is required for tensor parallel HPUGraph inference)
        # VLLM_HPU_COLLECTIVE_MARK_STEP=0 leaves it out, to measure the step without the graph cut.
        if gaudi_envs.VLLM_HPU_COLLECTIVE_MARK_STEP:
            htorch.core.mark_step()
        dist.all_reduce(input_, group=self.device_group)
        return input_

    def _all_reduce_as_gather(self, input_: torch.Tensor) -> torch.Tensor:
        """``VLLM_HPU_ALL_REDUCE_AS_GATHER_ROWS``: the sum of every rank's ``input_`` as an all-gather and a local sum.

        Every rank gathers all ``world`` partial tensors (7 times the payload on the wire against about 1.75 times
        for a ring all-reduce, one collective phase instead of two) and sums them itself in fp32, in the same op
        on every rank, so every rank holds the same bits. Out of place: ``input_`` is not written.
        """
        from vllm_gaudi import envs as gaudi_envs
        rows = input_.shape[0]
        gathered = torch.empty((self.world_size * rows, ) + tuple(input_.shape[1:]),
                               dtype=input_.dtype,
                               device=input_.device)
        if gaudi_envs.VLLM_HPU_COLLECTIVE_MARK_STEP:
            htorch.core.mark_step()
        dist.all_gather_into_tensor(gathered, input_.contiguous(), group=self.device_group)
        summed = gathered.view((self.world_size, ) + tuple(input_.shape)).sum(dim=0, dtype=torch.float32)
        return summed.to(input_.dtype)

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        world_size = self.world_size
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()
        input_size = input_.size()
        # Allocate output tensor.
        # NOTE: we have to use concat-style all-gather here,
        # stack-style all-gather has compatibility issues with
        # torch.compile . see https://github.com/pytorch/pytorch/issues/138795
        output_size = (input_size[0] * world_size, ) + input_size[1:]
        output_tensor = torch.empty(output_size, dtype=input_.dtype, device=input_.device)
        # All-gather.
        from vllm_gaudi import envs as gaudi_envs
        if gaudi_envs.VLLM_HPU_COLLECTIVE_MARK_STEP:
            htorch.core.mark_step()
        dist.all_gather_into_tensor(output_tensor, input_, group=self.device_group)
        # Reshape
        output_tensor = output_tensor.reshape((world_size, ) + input_size)
        output_tensor = output_tensor.movedim(0, dim)
        output_tensor = output_tensor.reshape(input_size[:dim] + (world_size * input_size[dim], ) +
                                              input_size[dim + 1:])
        return output_tensor

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if extra_tensors is not None:
            raise NotImplementedError("extra_tensors is not supported for HPU")
        # Use dispatch_tensor in the plugin FusedMoEMethod for better performance
        return hidden_states, router_logits

    def combine(self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False) -> torch.Tensor:
        if htorch.utils.internal.is_lazy():
            htorch.core.mark_step()
        assert self.dp_group is not None
        assert hidden_states.dim() == 2, "Input hidden states must be 2D"

        dp_metadata = get_hpu_dp_metadata()
        if dp_metadata is not None:
            local_hidden_states = dp_metadata.local_hidden_states
        else:
            local_num_tokens = hidden_states.size(0) // self.world_size if is_sequence_parallel else hidden_states.size(
                0) // self.dp_world_size
            local_hidden_states = torch.empty((local_num_tokens, hidden_states.size(-1)),
                                              device=hidden_states.device,
                                              dtype=hidden_states.dtype)

        torch.distributed.reduce_scatter_tensor(
            local_hidden_states,
            hidden_states,
            group=get_ep_group().device_group if is_sequence_parallel else self.dp_group.device_group)
        hidden_states = local_hidden_states
        return hidden_states
