import os
import numpy as np
import math

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp

from mindspore import log, Tensor, Parameter

from utils import config

mindspore.set_seed(0)
np.random.seed(0)

class Embeddings(nn.Cell):
    def __init__(self, vocab, d_model, padding_idx=None):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model, dtype=mindspore.float32, padding_idx=padding_idx)
        self.d_model = d_model

    def construct(self, x):
        return self.lut(x.astype(mindspore.int32)) * math.sqrt(self.d_model)

def gen_embeddings(vocab):
    """
        Generate an initial embedding matrix for `word_dict`.
        If an embedding file is not given or a word is not in the embedding file,
        a randomly initialized vector will be used.
    """
    embeddings = np.random.randn(vocab.n_words, config.emb_dim) * 0.01 
    print('Embeddings: %d x %d' % (vocab.n_words, config.emb_dim))
    if config.emb_file is not None:
        print('Loading embedding file: %s' % config.emb_file)
        pre_trained = 0
        for line in open(config.emb_file).readlines():
            sp = line.split()
            if(len(sp) == config.emb_dim + 1):
                if sp[0] in vocab.word2index:
                    pre_trained += 1
                    embeddings[vocab.word2index[sp[0]]] = [float(x) for x in sp[1:]]
            else:
                print(sp[0])
        print('Pre-trained: %d (%.2f%%)' % (pre_trained, pre_trained * 100.0 / vocab.n_words))
    return embeddings

def share_embedding(vocab, pretrain=True):
    embedding = Embeddings(vocab.n_words, config.emb_dim, padding_idx=config.PAD_idx)
    if(pretrain):
        pre_embedding = gen_embeddings(vocab)
        embedding.lut = nn.Embedding(vocab.n_words, config.emb_dim, embedding_table=Tensor(pre_embedding).astype(mindspore.float32), dtype=mindspore.float32, padding_idx=config.PAD_idx)
    return embedding

class MultiHeadAttention(nn.Cell):
    """
    Multi-head attention as per https://arxiv.org/pdf/1706.03762.pdf
    Refer Figure 2
    """
    def __init__(self, input_depth, total_key_depth, total_value_depth, output_depth, 
                 num_heads, bias_mask=None, dropout=0.0):
        """
        Parameters:
            input_depth: Size of last dimension of input
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            num_heads: Number of attention heads
            bias_mask: Masking tensor to prevent connections to future elements
            dropout: Dropout probability (Should be non-zero only during training)
        """
        super(MultiHeadAttention, self).__init__()
        # Checks borrowed from 
        # https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
        # if total_key_depth % num_heads != 0:
        #     raise ValueError("Key depth (%d) must be divisible by the number of "
        #                      "attention heads (%d)." % (total_key_depth, num_heads))
        # if total_value_depth % num_heads != 0:
        #     raise ValueError("Value depth (%d) must be divisible by the number of "
        #                      "attention heads (%d)." % (total_value_depth, num_heads))
            
        if total_key_depth % num_heads != 0:
            print("Key depth (%d) must be divisible by the number of "
                             "attention heads (%d)." % (total_key_depth, num_heads))
            total_key_depth = total_key_depth - (total_key_depth % num_heads)
        if total_value_depth % num_heads != 0:
            print("Value depth (%d) must be divisible by the number of "
                             "attention heads (%d)." % (total_value_depth, num_heads))
            total_value_depth = total_value_depth - (total_value_depth % num_heads)

        self.num_heads = num_heads
        self.query_scale = (total_key_depth//num_heads)**-0.5 ## sqrt
        self.bias_mask = bias_mask
        
        # Key and query depth will be same
        self.query_linear = nn.Dense(input_depth, total_key_depth, has_bias=False)
        self.key_linear = nn.Dense(input_depth, total_key_depth, has_bias=False)
        self.value_linear = nn.Dense(input_depth, total_value_depth, has_bias=False)
        self.output_linear = nn.Dense(total_value_depth, output_depth, has_bias=False)
        
        self.dropout = nn.Dropout(1 - dropout)
    
    def _split_heads(self, x):
        """
        Split x such to add an extra num_heads dimension
        Input:
            x: a Tensor with shape [batch_size, seq_length, depth]
        Returns:
            A Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        """
        if len(x.shape) != 3:
            raise ValueError("x must have rank 3")
        shape = x.shape

        return ops.transpose(x.view((shape[0], shape[1], self.num_heads, shape[2]//self.num_heads)), (0, 2, 1, 3))
    
    def _merge_heads(self, x):
        """
        Merge the extra num_heads into the last dimension
        Input:
            x: a Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        Returns:
            A Tensor with shape [batch_size, seq_length, depth]
        """
        if len(x.shape) != 4:
            raise ValueError("x must have rank 4")
        shape = x.shape
        return ops.transpose(x, (0, 2, 1, 3)).view((shape[0], shape[2], shape[3]*self.num_heads))
        
    def construct(self, queries, keys, values, mask):
        
        # Do a linear for each component
        queries = self.query_linear(queries.astype(mindspore.float32))
        keys = self.key_linear(keys.astype(mindspore.float32))
        values = self.value_linear(values.astype(mindspore.float32))
        
        # Split into multiple heads
        queries = self._split_heads(queries)
        keys = self._split_heads(keys)
        values = self._split_heads(values)
        
        # Scale queries
        queries *= self.query_scale
        
        # Combine queries and keys
        logits = ops.matmul(queries, ops.transpose(keys, (0, 1, 3, 2)))
        
        if mask is not None:
            mask = ops.expand_dims(mask, 1)  # [B, 1, 1, T_values]
            logits = logits.masked_fill(mask, -1e18)

        ## attention weights 
        attetion_weights = logits.sum(axis=1) / self.num_heads

        # Convert to probabilites
        weights = ops.softmax(logits, axis=-1)

        # Dropout
        weights = self.dropout(weights)
        
        # Combine with values to get context
        if len(weights.shape) < 3 and len(values.shape) < 3:
            contexts = ops.matmul(weights, values)
        else:
            contexts = ops.matmul(weights, values)
        
        # Merge heads
        contexts = self._merge_heads(contexts)
        
        # Linear to get output
        outputs = self.output_linear(contexts)
        
        return outputs, attetion_weights

class Conv(nn.Cell):
    """
    Convenience class that does padding and convolution for inputs in the format
    [batch_size, sequence length, hidden size]
    """
    def __init__(self, input_size, output_size, kernel_size, pad_type):
        """
        Parameters:
            input_size: Input feature size
            output_size: Output feature size
            kernel_size: Kernel width
            pad_type: left -> pad on the left side (to mask future data), 
                      both -> pad on both sides
        """
        super(Conv, self).__init__()
        padding = (kernel_size - 1, 0) if pad_type == 'left' else (kernel_size//2, (kernel_size - 1)//2)
        self.pad = nn.ConstantPad1d(padding, 0)
        self.conv = nn.Conv1d(input_size, output_size, kernel_size=kernel_size, padding=0)

    def construct(self, inputs):
        length = len(inputs[0])
        inputs = self.pad(ops.transpose(inputs, (0, 2, 1)).astype(mindspore.float32))[:, :, :length]
        outputs = ops.transpose(self.conv(inputs), (0, 2, 1))

        return outputs

class PositionwiseFeedForward(nn.Cell):
    """
    Does a Linear + RELU + Linear on each of the timesteps
    """
    def __init__(self, input_depth, filter_size, output_depth, layer_config='ll', padding='left', dropout=0.0):
        """
        Parameters:
            input_depth: Size of last dimension of input
            filter_size: Hidden size of the middle layer
            output_depth: Size last dimension of the final output
            layer_config: ll -> linear + ReLU + linear
                          cc -> conv + ReLU + conv etc.
            padding: left -> pad on the left side (to mask future data), 
                     both -> pad on both sides
            dropout: Dropout probability (Should be non-zero only during training)
        """
        super(PositionwiseFeedForward, self).__init__()
        
        layers = []
        sizes = ([(input_depth, filter_size)] + 
                 [(filter_size, filter_size)]*(len(layer_config)-2) + 
                 [(filter_size, output_depth)])

        for lc, s in zip(list(layer_config), sizes):
            if lc == 'l':
                layers.append(nn.Dense(*s))
            elif lc == 'c':
                layers.append(Conv(*s, kernel_size=3, pad_type=padding))
            else:
                raise ValueError("Unknown layer type {}".format(lc))

        self.layers = nn.CellList(layers)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(1 - dropout)
        
    def construct(self, inputs):
        x = inputs
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers):
                x = self.relu(x)
                x = self.dropout(x)

        return x

class EncoderLayer(nn.Cell):
    """
    Represents one Encoder layer of the Transformer Encoder
    Refer Fig. 1 in https://arxiv.org/pdf/1706.03762.pdf
    NOTE: The layer normalization step has been moved to the input as per latest version of T2T
    """
    def __init__(self, hidden_size, total_key_depth, total_value_depth, filter_size, num_heads,
                 bias_mask=None, layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0):
        """
        Parameters:
            hidden_size: Hidden size
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            num_heads: Number of attention heads
            bias_mask: Masking tensor to prevent connections to future elements
            layer_dropout: Dropout for this layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """
        super(EncoderLayer, self).__init__()
        
        self.multi_head_attention = MultiHeadAttention(hidden_size, total_key_depth, total_value_depth, 
                                                       hidden_size, num_heads, bias_mask, attention_dropout)
        
        self.positionwise_feed_forward = PositionwiseFeedForward(hidden_size, filter_size, hidden_size,
                                                                 layer_config='cc', padding = 'both', 
                                                                 dropout=relu_dropout)
        self.dropout = nn.Dropout(1 - layer_dropout)
        self.layer_norm_mha = nn.LayerNorm((hidden_size,))
        self.layer_norm_ffn = nn.LayerNorm((hidden_size,))
        
    def construct(self, inputs, mask=None):
        x = inputs
        
        # Layer Normalization
        x_norm = self.layer_norm_mha(x.astype(mindspore.float32))
        
        # Multi-head attention
        y, _ = self.multi_head_attention(x_norm, x_norm, x_norm, mask)
        
        # Dropout and residual
        x = self.dropout(x + y)
        
        # Layer Normalization
        x_norm = self.layer_norm_ffn(x.astype(mindspore.float32))

        # Positionwise Feedforward
        y = self.positionwise_feed_forward(x_norm)

        # Dropout and residual
        y = self.dropout(x + y)
        
        # y = self.layer_norm_end(y)
        return y

class DecoderLayer(nn.Cell):
    """
    Represents one Decoder layer of the Transformer Decoder
    Refer Fig. 1 in https://arxiv.org/pdf/1706.03762.pdf
    NOTE: The layer normalization step has been moved to the input as per latest version of T2T
    """
    def __init__(self, hidden_size, total_key_depth, total_value_depth, filter_size, num_heads,
                 bias_mask, layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0):
        """
        Parameters:
            hidden_size: Hidden size
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            num_heads: Number of attention heads
            bias_mask: Masking tensor to prevent connections to future elements
            layer_dropout: Dropout for this layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """
        super(DecoderLayer, self).__init__()
        
        self.multi_head_attention_dec = MultiHeadAttention(hidden_size, total_key_depth, total_value_depth, 
                                                       hidden_size, num_heads, bias_mask, attention_dropout)

        self.multi_head_attention_enc_dec = MultiHeadAttention(hidden_size, total_key_depth, total_value_depth, 
                                                       hidden_size, num_heads, None, attention_dropout)
        
        self.positionwise_feed_forward = PositionwiseFeedForward(hidden_size, filter_size, hidden_size,
                                                                 layer_config='cc', padding = 'left', 
                                                                 dropout=relu_dropout)
        self.dropout = nn.Dropout(1 - layer_dropout)
        self.layer_norm_mha_dec = nn.LayerNorm((hidden_size,))
        self.layer_norm_mha_enc = nn.LayerNorm((hidden_size,))
        self.layer_norm_ffn = nn.LayerNorm((hidden_size,))
        # self.layer_norm_end = LayerNorm(hidden_size)

    def construct(self, inputs):
        """
        NOTE: Inputs is a tuple consisting of decoder inputs and encoder output
        """

        x, encoder_outputs, attention_weight, mask = inputs
        mask_src, dec_mask = mask
        
        # Layer Normalization before decoder self attention
        x_norm = self.layer_norm_mha_dec(x.astype(mindspore.float32))
        
        # Masked Multi-head attention
        y, _ = self.multi_head_attention_dec(x_norm, x_norm, x_norm, dec_mask)
        
        # Dropout and residual after self-attention
        x = self.dropout(x + y)

        # Layer Normalization before encoder-decoder attention
        x_norm = self.layer_norm_mha_enc(x.astype(mindspore.float32))

        # Multi-head encoder-decoder attention
        y, attention_weight = self.multi_head_attention_enc_dec(x_norm, encoder_outputs, encoder_outputs, mask_src)

        # Dropout and residual after encoder-decoder attention
        x = self.dropout(x + y)
        
        # Layer Normalization
        x_norm = self.layer_norm_ffn(x.astype(mindspore.float32))
        
        # Positionwise Feedforward
        y = self.positionwise_feed_forward(x_norm)
        
        # Dropout and residual after positionwise feed forward layer
        y = self.dropout(x + y)
        
        # y = self.layer_norm_end(y)
        
        # Return encoder outputs as well to work with nn.Sequential
        return y, encoder_outputs, attention_weight, mask

def _gen_timing_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    """
    Generates a [1, length, channels] timing signal consisting of sinusoids
    Adapted from:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
    """
    position = np.arange(length)
    num_timescales = channels // 2
    log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
    inv_timescales = min_timescale * np.exp(np.arange(num_timescales).astype(np.float) * -log_timescale_increment)
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)

    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, channels % 2]], 'constant', constant_values=[0.0, 0.0])
    signal =  signal.reshape([1, length, channels])

    return Tensor.from_numpy(signal).astype(mindspore.float32)

def _gen_bias_mask(max_length):
    """
    Generates bias values (-Inf) to mask future timesteps during attention
    """
    np_mask = np.triu(np.full([max_length, max_length], -np.inf), 1)
    mindspore_mask = Tensor.from_numpy(np_mask).astype(mindspore.float32)

    bias_mask = ops.expand_dims(ops.expand_dims(mindspore_mask, 0), 1)
    
    return bias_mask

def _get_attn_subsequent_mask(size):
    """
    Get an attention mask to avoid using the subsequent info.
    Args:
        size: int
    Returns:
        (`LongTensor`):
        * subsequent_mask `[1 x size x size]`
    """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = Tensor.from_numpy(subsequent_mask)

    return subsequent_mask


