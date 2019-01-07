
import numpy as np
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import PackedSequence

from lib.fpn.box_utils import nms_overlaps
from lib.word_vectors import obj_edge_vectors
from .highway_lstm_cuda.alternating_highway_lstm import block_orthogonal

from IPython import embed


def get_dropout_mask(
        dropout_probability: float,
        tensor_for_masking: torch.autograd.Variable
):
    """Computes and returns an element-wise dropout mask for a given tensor, where
    each element in the mask is dropped out with probability dropout_probability.
    Note that the mask is NOT applied to the tensor - the tensor is passed to retain
    the correct CUDA tensor type for the mask.
    Args:
        dropout_probability: float, required.
            Probability of dropping a dimension of the input.
        tensor_for_masking: torch.Variable, required.
    Returns:
        dropout_mask: torch.FloatTensor, consisting of the binary mask scaled by 1/ (1 - dropout_probability).
            This scaling ensures expected values and variances of the output of applying this mask
            and the original tensor are the same.
    """
    binary_mask = tensor_for_masking.clone()
    binary_mask.data.copy_(torch.rand(tensor_for_masking.size()) > dropout_probability)
    # Scale mask by 1/keep_prob to preserve output statistics.
    dropout_mask = binary_mask.float().div(1.0 - dropout_probability)
    return dropout_mask


class MemoryRNN(torch.nn.Module):
    def __init__(
            self,
            classes,
            rel_classes,
            inputs_dim,
            hidden_dim,
            recurrent_dropout_probability=0.2,
            use_highway=True,
            use_input_projection_bias=True
    ):
        """Initializes the RNN
        Args:
            classes:
            rel_classes:
            inputs_dim:
            hidden_dim: Hidden dim of the decoder
            recurrent_dropout_probability:
            use_highway:
            use_input_projection_bias:
        """
        # TODO add database bias in this module
        super(MemoryRNN, self).__init__()

        self.classes = classes
        self.rel_classes = rel_classes
        self.hidden_size = hidden_dim
        self.inputs_dim = inputs_dim
        self.nms_thresh = 0.3

        self.rel_mem_h = nn.Embedding(self.num_rels, hidden_dim)
        self.rel_mem_h.weight.data.fill_(0)
        self.rel_mem_c = nn.Embedding(self.num_rels, hidden_dim)
        self.rel_mem_c.weight.data.fill_(0)

        self.recurrent_dropout_probability = recurrent_dropout_probability
        self.use_highway = use_highway

        # We do the projections for all the gates all at once, so if we are
        # using highway layers, we need some extra projections, which is
        # why the sizes of the Linear layers change here depending on this flag.
        if use_highway:
            self.input_linearity = torch.nn.Linear(
                self.inputs_dim, 6 * self.hidden_size,
                bias=use_input_projection_bias
            )
            self.state_linearity = torch.nn.Linear(
                self.hidden_size, 5 * self.hidden_size,
                bias=True
            )
        else:
            self.input_linearity = torch.nn.Linear(
                self.inputs_dim, 4 * self.hidden_size,
                bias=use_input_projection_bias
            )
            self.state_linearity = torch.nn.Linear(
                self.hidden_size, 4 * self.hidden_size,
                bias=True
            )

        self.out = nn.Linear(self.hidden_size, len(self.rel_classes))
        self.reset_parameters()

    @property
    def num_rels(self):
        return len(self.rel_classes)

    @property
    def input_size(self):
        return self.inputs_dim

    def reset_parameters(self):
        # Use sensible default initializations for parameters.
        block_orthogonal(self.input_linearity.weight.data, [self.hidden_size, self.input_size])
        block_orthogonal(self.state_linearity.weight.data, [self.hidden_size, self.hidden_size])

        self.state_linearity.bias.data.fill_(0.0)
        # Initialize forget gate biases to 1.0 as per An Empirical
        # Exploration of Recurrent Network Architectures, (Jozefowicz, 2015).
        self.state_linearity.bias.data[self.hidden_size:2 * self.hidden_size].fill_(1.0)

    def lstm_equations(
            self,
            timestep_input,
            previous_state,
            previous_memory,
            dropout_mask=None
    ):
        """Does the hairy LSTM math
        Args:
            timestep_input:
            previous_state:
            previous_memory:
            dropout_mask:
        Returns:
        """
        # Do the projections for all the gates all at once.
        projected_input = self.input_linearity(timestep_input)
        projected_state = self.state_linearity(previous_state)

        # Main LSTM equations using relevant chunks of the big linear
        # projections of the hidden state and inputs.
        input_gate = torch.sigmoid(
            projected_input[:, 0 * self.hidden_size:1 * self.hidden_size] +
            projected_state[:, 0 * self.hidden_size:1 * self.hidden_size]
        )
        forget_gate = torch.sigmoid(
            projected_input[:, 1 * self.hidden_size:2 * self.hidden_size] +
            projected_state[:, 1 * self.hidden_size:2 * self.hidden_size]
        )
        memory_init = torch.tanh(
            projected_input[:, 2 * self.hidden_size:3 * self.hidden_size] +
            projected_state[:, 2 * self.hidden_size:3 * self.hidden_size]
        )
        output_gate = torch.sigmoid(
            projected_input[:, 3 * self.hidden_size:4 * self.hidden_size] +
            projected_state[:, 3 * self.hidden_size:4 * self.hidden_size]
        )
        memory = input_gate * memory_init + forget_gate * previous_memory
        timestep_output = output_gate * torch.tanh(memory)

        if self.use_highway:
            highway_gate = torch.sigmoid(
                projected_input[:, 4 * self.hidden_size:5 * self.hidden_size] +
                projected_state[:, 4 * self.hidden_size:5 * self.hidden_size]
            )
            highway_input_projection = projected_input[:, 5 * self.hidden_size:6 * self.hidden_size]
            timestep_output = highway_gate * timestep_output + (1 - highway_gate) * highway_input_projection

        # Only do dropout if the dropout prob is > 0.0 and we are in training mode.
        if dropout_mask is not None and self.training:
            timestep_output = timestep_output * dropout_mask
        return timestep_output, memory

    def forward(
            self,  # pylint: disable=arguments-differ
            inputs: PackedSequence,
            initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            rel_labels: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            inputs : PackedSequence, required.
            A tensor of shape (batch_size, num_timesteps, input_size)
            to apply the LSTM over.

            initial_state : Tuple[torch.Tensor, torch.Tensor], optional, (default = None)
            A tuple (state, memory) representing the initial hidden state and memory
            of the LSTM. Each tensor has shape (1, batch_size, output_dimension).

            rel_labels: Variable, the relation label, (NumOfRels, 4)
                e.g. labels[0,:] = [imgid, box0, box1, rel_class_id]
        Returns:
            out_dists:
                in training mode: PackSequence, it contains a torch.FloatTensor of shape
                    (NumOfRels, output_dimension) representing the outputs of the LSTM per timestep
                in test mode: List of PackSequence, it contains output based on each relation memory.
                    the length of the list is #Rel.
                    e.g. out_dists[0] = PackSequence with shape of (NumOfRels, Output_dimemsion)
        """
        if not isinstance(inputs, PackedSequence):
            raise ValueError('inputs must be PackedSequence but got %s' % (type(inputs)))
        if self.training:
            assert rel_labels is not None, 'Relation labels should be provided to train this module'

        sequence_tensor, batch_lengths = inputs
        batch_size = batch_lengths[0]
        nr_rels = rel_labels.size(0)

        # We're just doing an LSTM decoder here so ignore states, etc
        if initial_state is None:
            previous_memory = Variable(
                sequence_tensor.data.new().resize_(batch_size, self.hidden_size).fill_(0)
            )
            previous_state = Variable(
                sequence_tensor.data.new().resize_(batch_size, self.hidden_size).fill_(0)
            )
        else:
            assert len(initial_state) == 2, 'Inital_state should contains h0, c0 of lstm'
            previous_state = initial_state[0].squeeze(0)
            previous_memory = initial_state[1].squeeze(0)

        if self.recurrent_dropout_probability > 0.0:
            dropout_mask = get_dropout_mask(self.recurrent_dropout_probability, previous_memory)
        else:
            dropout_mask = None

        # Only accumulating label predictions here, discarding everything else
        if self.training:
            out_dists = []
        else:
            out_dists = [[] for _ in range(nr_rels)]

        end_ind = 0
        for i, l_batch in enumerate(batch_lengths):
            start_ind = end_ind
            end_ind = end_ind + l_batch

            if previous_memory.size(0) != l_batch:
                previous_memory = previous_memory[:l_batch]
                previous_state = previous_state[:l_batch]
                if dropout_mask is not None:
                    dropout_mask = dropout_mask[:l_batch]

            if self.training:
                # load relation class memory
                for offset in range(l_batch):
                    ind = start_ind + offset
                    previous_memory[offset].data = self.rel_mem_c(rel_labels[ind][3]).data
                    previous_state[offset].data = self.rel_mem_h(rel_labels[ind][3]).data

                timestep_input = sequence_tensor[start_ind:end_ind]

                previous_state, previous_memory = self.lstm_equations(
                    timestep_input,
                    previous_state,
                    previous_memory,
                    dropout_mask=dropout_mask
                )

                # store relation class memory
                for offset in range(l_batch):
                    ind = start_ind + offset
                    self.rel_mem_c.weight.data[rel_labels[ind][3].data] = previous_memory[offset].data[None, :]
                    self.rel_mem_h.weight.data[rel_labels[ind][3].data] = previous_state[offset].data[None, :]

                pred_dist = self.out(previous_state)
                out_dists.append(pred_dist)
            else:  # test mode
                for rel_i in range(self.num_rels):
                    for offset in range(l_batch):
                        previous_memory[offset].data = self.rel_mem_c(rel_i).data
                        previous_state[offset].data = self.rel_mem_h(rel_i).data
                    timestep_input = sequence_tensor[start_ind:end_ind]

                    previous_state, previous_memory = self.lstm_equations(
                        timestep_input,
                        previous_state,
                        previous_memory,
                        dropout_mask=dropout_mask
                    )

                    pred_dist = self.out(previous_state)
                    out_dists[rel_i].append(pred_dist)

        if self.training:
            out_dists = torch.cat(out_dists, 0)
            out_dists = PackedSequence(out_dists, batch_lengths)
        else:
            for rel_i in range(self.num_rels):
                out_dists[rel_i] = torch.cat(out_dists[rel_i], 0)
                out_dists = PackedSequence(out_dists, batch_lengths)

        return out_dists
