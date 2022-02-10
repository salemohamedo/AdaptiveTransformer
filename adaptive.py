# from collections import namedtuple
#
# import torch
# from torch import nn
# import torch.nn.functional as F
#
# AdaptiveSoftmaxOutput = namedtuple('AdaptiveSoftmaxOutput', ['output', 'loss'])
#
#
# class AdaptiveTail(nn.Module):
#     def __init__(self, ndim, ntoken, cutoffs, div_value=4):
#         super(AdaptiveTail, self).__init__()
#         self.div_value = div_value
#         self.ndim = ndim
#         self.cutoffs = cutoffs + [ntoken]
#         self.tail_clusters = nn.ModuleList()
#         for i, l_bound in enumerate(self.cutoffs[:-1]):
#             cluster_size = self.cutoffs[i + 1] - l_bound
#             self.tail_clusters.append(
#                 nn.Sequential(
#                     nn.Embedding(cluster_size, ndim // (div_value ** (i + 1))),
#                     nn.Linear(ndim // (div_value ** (i + 1)), self.ndim, bias=False)
#                 )
#             )
#
#         def init_weights(m):
#             if isinstance(m, nn.Embedding):
#                 nn.init.normal_(m.weight, mean=0, std=m.weight.shape[1] ** -0.5)
#             elif hasattr(m, "weight"):
#                 nn.init.xavier_uniform_(m.weight)
#
#         self.apply(init_weights)
#
#     def forward(self, inputs, cluster, softmax=True):
#         if softmax:
#             outputs = F.linear(inputs, self.tail_clusters[cluster][1].weight.T)
#             return F.linear(outputs, self.tail_clusters[cluster][0].weight)
#         else:
#             return self.tail_clusters[cluster](inputs)
#
#
# class AdaptiveSoftmax(nn.Module):
#     def __init__(self, ndim, ntoken, cutoffs, div_value=4, shared_tail=None):
#         super(AdaptiveSoftmax, self).__init__()
#         self.div_value = div_value
#         self.ndim = ndim
#         self.cutoffs = cutoffs + [ntoken]
#         self.head_size = self.cutoffs[0] + len(self.cutoffs) - 1
#
#         self.head_cluster = nn.Linear(self.ndim, self.head_size, bias=False)
#         nn.init.xavier_uniform_(self.head_cluster.weight)
#
#         if shared_tail is not None:
#             self.tail_clusters = shared_tail
#         else:
#             self.tail_clusters = AdaptiveTail(ndim, ntoken, cutoffs, div_value)
#
#     def map_target_to_cluster(self, targets):
#         cluster_targets = []
#         head_targets = targets.clone()
#         for i in range(len(self.cutoffs) - 1):
#             l_bound = self.cutoffs[i]
#             u_bound = self.cutoffs[i + 1]
#             targets_in_range = targets.ge(l_bound).logical_and(targets.lt(u_bound))
#             targets_in_range = targets_in_range.nonzero().squeeze(dim=1)
#             cluster_targets.append(targets_in_range)
#             head_targets[targets_in_range] = self.cutoffs[0] + i
#         return cluster_targets, head_targets
#
#     def forward(self, inputs, targets):
#         outputs = inputs.new_zeros(targets.size(0))
#
#         cluster_targets, head_targets = self.map_target_to_cluster(targets)
#         head_output = self.head_cluster(inputs)
#         head_output = head_output.log_softmax(dim=1)
#         head_output = head_output.gather(1, head_targets.unsqueeze(1))
#         outputs += head_output.squeeze()
#
#         for i, ids in enumerate(cluster_targets):
#             if len(ids) == 0:  # no targets for this cluster
#                 continue
#             cluster_outputs = self.tail_clusters(inputs[ids], i, softmax=True)
#             cluster_outputs = cluster_outputs.log_softmax(dim=1)
#             relative_targets = targets[ids] - self.cutoffs[i]
#             cluster_outputs = cluster_outputs.gather(1, relative_targets.unsqueeze(1))
#             outputs[ids] += cluster_outputs.squeeze()
#
#         loss = (-outputs).mean()
#         return AdaptiveSoftmaxOutput(outputs, loss)
#
#
# class AdaptiveInput(nn.Module):
#     def __init__(self, ndim, ntoken, cutoffs, div_value=4, shared_tail=None):
#         super(AdaptiveInput, self).__init__()
#         self.div_value = div_value
#         self.ndim = ndim
#         self.cutoffs = cutoffs + [ntoken]
#         self.head_size = self.cutoffs[0] + len(self.cutoffs) - 1
#
#         self.head_cluster = nn.Sequential(
#             nn.Embedding(self.cutoffs[0], self.ndim),
#             nn.Linear(self.ndim, self.ndim)
#         )
#         nn.init.normal_(self.head_cluster[0].weight, mean=0, std=self.head_cluster[0].weight.shape[1] ** -0.5)
#         nn.init.xavier_uniform_(self.head_cluster[1].weight)
#         if shared_tail is not None:
#             self.tail_clusters = shared_tail
#         else:
#             self.tail_clusters = AdaptiveTail(ndim, ntoken, cutoffs, div_value)
#
#     def forward(self, inputs):
#         outputs = inputs.new_zeros(inputs.shape + (self.ndim,), dtype=torch.float)
#         cutoffs = [0] + self.cutoffs
#         for i in range(len(cutoffs) - 1):
#             l_bound = cutoffs[i]
#             u_bound = cutoffs[i + 1]
#             cluster_mask = inputs.ge(l_bound).logical_and(inputs.lt(u_bound))
#             cluster_inputs = inputs[cluster_mask] - cutoffs[i]
#             if i == 0:
#                 cluster_output = self.head_cluster(cluster_inputs)
#             else:
#                 cluster_output = self.tail_clusters(cluster_inputs, i - 1, softmax=False)
#             outputs[cluster_mask] = cluster_output
#         return outputs
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


def quant_noise(module, p, block_size):
    """
    Wraps modules and applies quantization noise to the weights for
    subsequent quantization with Iterative Product Quantization as
    described in "Training with Quantization Noise for Extreme Model Compression"

    Args:
        - module: nn.Module
        - p: amount of Quantization Noise
        - block_size: size of the blocks for subsequent quantization with iPQ

    Remarks:
        - Module weights must have the right sizes wrt the block size
        - Only Linear, Embedding and Conv2d modules are supported for the moment
        - For more detail on how to quantize by blocks with convolutional weights,
          see "And the Bit Goes Down: Revisiting the Quantization of Neural Networks"
        - We implement the simplest form of noise here as stated in the paper
          which consists in randomly dropping blocks
    """

    # if no quantization noise, don't register hook
    if p <= 0:
        return module

    # supported modules
    assert isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d))

    # test whether module.weight has the right sizes wrt block_size
    is_conv = module.weight.ndim == 4

    # 2D matrix
    if not is_conv:
        assert (
            module.weight.size(1) % block_size == 0
        ), "Input features must be a multiple of block sizes"

    # 4D matrix
    else:
        # 1x1 convolutions
        if module.kernel_size == (1, 1):
            assert (
                module.in_channels % block_size == 0
            ), "Input channels must be a multiple of block sizes"
        # regular convolutions
        else:
            k = module.kernel_size[0] * module.kernel_size[1]
            assert k % block_size == 0, "Kernel size must be a multiple of block size"

    def _forward_pre_hook(mod, input):
        # no noise for evaluation
        if mod.training:
            if not is_conv:
                # gather weight and sizes
                weight = mod.weight
                in_features = weight.size(1)
                out_features = weight.size(0)

                # split weight matrix into blocks and randomly drop selected blocks
                mask = torch.zeros(
                    in_features // block_size * out_features, device=weight.device
                )
                mask.bernoulli_(p)
                mask = mask.repeat_interleave(block_size, -1).view(-1, in_features)

            else:
                # gather weight and sizes
                weight = mod.weight
                in_channels = mod.in_channels
                out_channels = mod.out_channels

                # split weight matrix into blocks and randomly drop selected blocks
                if mod.kernel_size == (1, 1):
                    mask = torch.zeros(
                        int(in_channels // block_size * out_channels),
                        device=weight.device,
                    )
                    mask.bernoulli_(p)
                    mask = mask.repeat_interleave(block_size, -1).view(-1, in_channels)
                else:
                    mask = torch.zeros(
                        weight.size(0), weight.size(1), device=weight.device
                    )
                    mask.bernoulli_(p)
                    mask = (
                        mask.unsqueeze(2)
                        .unsqueeze(3)
                        .repeat(1, 1, mod.kernel_size[0], mod.kernel_size[1])
                    )

            # scale weights and apply mask
            mask = mask.to(
                torch.bool
            )  # x.bool() is not currently supported in TorchScript
            s = 1 / (1 - p)
            mod.weight.data = s * weight.masked_fill(mask, 0)

    module.register_forward_pre_hook(_forward_pre_hook)
    return module

from typing import List

import torch
from torch import nn

# from fairseq.modules.quant_noise import quant_noise


class AdaptiveInput(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        padding_idx: int,
        initial_dim: int,
        factor: float,
        output_dim: int,
        cutoff: List[int],
        q_noise: float = 0,
        qn_block_size: int = 8,
    ):
        super().__init__()

        if vocab_size > cutoff[-1]:
            cutoff = cutoff + [vocab_size]
        else:
            assert (
                vocab_size == cutoff[-1]
            ), "cannot specify cutoff larger than vocab size"

        self.cutoff = cutoff
        self.embedding_dim = output_dim
        self.padding_idx = padding_idx

        self.embeddings = nn.ModuleList()
        for i in range(len(self.cutoff)):
            prev = self.cutoff[i - 1] if i > 0 else 0
            size = self.cutoff[i] - prev
            dim = int(initial_dim // (factor**i))
            seq = nn.Sequential(
                nn.Embedding(size, dim),
                quant_noise(
                    nn.Linear(dim, output_dim, bias=False), q_noise, qn_block_size
                ),
            )

            self.embeddings.append(seq)
            self.padding_idx = None
        self.padding_idx = padding_idx

        def init_weights(m):
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=m.weight.shape[1] ** -0.5)
                # nn.init.constant_(m.weight[padding_idx], 0)
            elif hasattr(m, "weight"):
                nn.init.xavier_uniform_(m.weight)

        self.apply(init_weights)

        self.register_buffer("_float_tensor", torch.FloatTensor(1))

    def weights_for_band(self, band: int):
        return self.embeddings[band][0].weight, self.embeddings[band][1].weight

    def forward(self, input: torch.Tensor):
        result = self._float_tensor.new(input.shape + (self.embedding_dim,))
        for i in range(len(self.cutoff)):
            mask = input.lt(self.cutoff[i])
            if i > 0:
                mask.mul_(input.ge(self.cutoff[i - 1]))
                chunk_input = input[mask] - self.cutoff[i - 1]
            else:
                chunk_input = input[mask]
            if mask.any():
                result[mask] = self.embeddings[i](chunk_input)
        return result