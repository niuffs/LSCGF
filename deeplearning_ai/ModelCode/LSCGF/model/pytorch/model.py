import torch
import torch.nn as nn
from torch.nn import functional as F
from model.pytorch.cell import DCGRUCell, SmoothSparseUnit
import numpy as np
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
minm = -99999999
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def cosine_similarity_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

class Adjacency_generator(nn.Module):
    def __init__(self, embedding_size, num_nodes, time_series, kernel_size, freq, requires_graph, seq_len, input_dim, device):
        super(Adjacency_generator, self).__init__()
        self.freq = freq
        self.kernel_size = kernel_size
        self.num_nodes = num_nodes
        self.embedding = embedding_size
        self.time_series = time_series
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.segm = int((self.time_series.shape[0]-1) // self.freq)
        self.graphs = requires_graph
        self.delta_series = torch.zeros(self.time_series.shape[0], self.time_series.shape[1]).to(device)
        self.conv1d = nn.Conv1d(in_channels=self.segm, out_channels=self.graphs, kernel_size=kernel_size, padding=0)
        self.fc_1 = nn.Linear(self.freq - self.kernel_size + 1, self.embedding)
        self.fc_2 = nn.Linear(self.embedding, self.num_nodes)
        self.process()
        self.device = device


    def process(self):

        for i in range(self.time_series.shape[0]):
            if i == 0:
                self.delta_series[i] = self.time_series[i]
            else:
                self.delta_series[i] = self.time_series[i]-self.time_series[i-1]
        times = []
        for i in range(self.segm):
            time_seg = self.delta_series[i * self.freq + 1:(i+1) * self.freq + 1]
            times.append(time_seg)

        self.times = torch.stack(times, dim=0) # (graphs, freq, num_nodes)

    def forward(self, node_feas): # input: (seq_len, batch_size, num_sensor * input_dim)
        mid_input = self.conv1d(self.times.permute(2,0,1)).permute(1,0,2) #(graphs, num_nodes, freq-kernel_size+1)
        mid_output = torch.stack([F.relu(self.fc_1(mid_input[i,...])) for i in range(self.graphs)], dim=0)
        output = torch.sigmoid(self.fc_2(mid_output))
        # 对于cos_similarity来说，值越大表示越接近
        max_similarity = minm
        seq_len = node_feas.shape[0]
        batch_size = node_feas.shape[1]
        node_feas = node_feas.reshape(seq_len, batch_size, self.num_nodes, -1)
        node_feas = node_feas.permute(1, 2, 3, 0)
        node_feas = node_feas.reshape(batch_size, self.num_nodes, -1)
        nodes_feature = torch.zeros(node_feas.shape[1], node_feas.shape[2]).to(self.device)
        for i in range(node_feas.shape[0]):
            nodes_feature += node_feas[i, ...]
        node_feas = torch.matmul(nodes_feature, nodes_feature.T)
        select = -1
        for graph_idx in range(output.shape[0]):
            x_1 = node_feas.reshape(1, -1)
            x_2 = output[graph_idx, :, :].reshape(1, -1)
            similarity = cosine_similarity_torch(x_1, x_2, eps=1e-20)
            if similarity > max_similarity:
                max_similarity = similarity
                select = graph_idx

        return output[select]

class Seq2SeqAttrs:
    def __init__(self, args):
        #self.adj_mx = adj_mx
        self.max_diffusion_step = args.max_diffusion_step
        self.cl_decay_steps = args.cl_decay_steps
        self.filter_type = args.filter_type
        self.num_nodes = args.num_nodes
        self.num_rnn_layers = args.num_rnn_layers
        self.rnn_units = args.rnn_units
        self.hidden_state_size = self.num_nodes * self.rnn_units


class EncoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, args):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, args)
        self.input_dim = args.input_dim
        self.seq_len = args.seq_len  # for the encoder
        self.device = args.device
        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(self.rnn_units, self.max_diffusion_step, self.num_nodes,
                       filter_type=self.filter_type, device=self.device) for _ in range(self.num_rnn_layers)])


    def forward(self, inputs, adj, hidden_state=None):
        """
        Encoder forward pass.
        :param inputs: shape (batch_size, self.num_nodes * self.input_dim)
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.hidden_state_size)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """
        batch_size, _ = inputs.size()
        if hidden_state is None:
            hidden_state = torch.zeros((self.num_rnn_layers, batch_size, self.hidden_state_size),
                                       device=self.device)
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num], adj)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        return output, torch.stack(hidden_states)  # runs in O(num_layers) so not too slow


class DecoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, args):
        # super().__init__(is_training, adj_mx, **model_kwargs)
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, args)
        self.output_dim = args.output_dim
        self.horizon = args.horizon  # for the decoder
        self.projection_layer = nn.Linear(self.rnn_units, self.output_dim)
        self.device = args.device
        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(self.rnn_units, self.max_diffusion_step, self.num_nodes,
                       filter_type=self.filter_type, device=self.device) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs, adj, hidden_state=None):
        """
        :param inputs: shape (batch_size, self.num_nodes * self.output_dim)
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.num_nodes * self.output_dim)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num], adj)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        projected = self.projection_layer(output.view(-1, self.rnn_units))
        output = projected.view(-1, self.num_nodes * self.output_dim)

        return output, torch.stack(hidden_states)


class LSCGFModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, node_feas, temperature, logger, args):
        super().__init__()
        Seq2SeqAttrs.__init__(self, args)
        self.args = args
        self.encoder_model = EncoderModel(args)
        self.decoder_model = DecoderModel(args)
        self.cl_decay_steps = args.cl_decay_steps
        self.use_curriculum_learning = args.use_curriculum_learning
        self._logger = logger
        self.temperature = temperature
        self.embedding_size = args.embedding_size
        self.seq_len = args.seq_len
        self.input_dim = args.input_dim
        self.kernel_size = args.kernel_size
        self.freq = args.freq
        self.requires_graph = args.requires_graph
        self.Adjacency_generator = Adjacency_generator(embedding_size=self.embedding_size, num_nodes = self.num_nodes, time_series = node_feas,
                                                       kernel_size=self.kernel_size, freq=self.freq, requires_graph = self.requires_graph,
                                                       seq_len=self.seq_len, input_dim = self.input_dim, device = args.device)
        self.device = args.device


    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def encoder(self, inputs, adj):
        """
        Encoder forward pass
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :return: encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        """
        encoder_hidden_state = None
        for t in range(self.args.seq_len):
            _, encoder_hidden_state = self.encoder_model(inputs[t], adj, encoder_hidden_state)

        return encoder_hidden_state

    def decoder(self, encoder_hidden_state, adj, labels=None, batches_seen=None):
        """
        Decoder forward pass
        :param encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        :param labels: (self.horizon, batch_size, self.num_nodes * self.output_dim) [optional, not exist for inference]
        :param batches_seen: global step [optional, not exist for inference]
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros((batch_size, self.num_nodes * self.decoder_model.output_dim),
                                device=self.device)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        outputs = []

        for t in range(self.decoder_model.horizon):
            decoder_output, decoder_hidden_state = self.decoder_model(decoder_input, adj,
                                                                      decoder_hidden_state)
            decoder_input = decoder_output
            outputs.append(decoder_output)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = labels[t]
        outputs = torch.stack(outputs)
        return outputs

    def forward(self, inputs, temp, labels=None, batches_seen=None):
        """
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :param labels: shape (horizon, batch_size, num_sensor * output)
        :param batches_seen: batches seen till now
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """

        adj = SmoothSparseUnit(self.Adjacency_generator(inputs), 1, 0.10)
        # adj = F.relu(self.Adjacency_generator(inputs))

        encoder_hidden_state = self.encoder(inputs, adj)
        self._logger.debug("Encoder complete, starting decoder")
        outputs = self.decoder(encoder_hidden_state, adj, labels, batches_seen=batches_seen)
        self._logger.debug("Decoder complete")
        if batches_seen == 0:
            self._logger.info(
                "Total trainable parameters {}".format(count_parameters(self))
            )

        return outputs, adj