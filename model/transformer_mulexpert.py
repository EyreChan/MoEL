import os
import random
import math
import numpy as np

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp

from mindspore import log, Tensor

from model.common_layer import EncoderLayer, DecoderLayer, share_embedding, \
    _gen_timing_signal, _gen_bias_mask, _get_attn_subsequent_mask
from utils import config

mindspore.set_seed(0)
np.random.seed(0)

class Encoder(nn.Cell):
    """
    A Transformer Encoder module. 
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """
    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=1000, input_dropout=0.0, layer_dropout=0.0, 
                 attention_dropout=0.0, relu_dropout=0.0, use_mask=False, universal=False):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
            use_mask: Set to True to turn on future value masking
        """
        
        super(Encoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)
        
        if(self.universal):  
            ## for t
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        params = (hidden_size, 
                 total_key_depth or hidden_size,
                 total_value_depth or hidden_size,
                 filter_size, 
                 num_heads, 
                 _gen_bias_mask(max_length) if use_mask else None,
                 layer_dropout, 
                 attention_dropout, 
                 relu_dropout)
        
        self.embedding_proj = nn.Dense(embedding_size, hidden_size, has_bias=False)
        if(self.universal):
            self.enc = EncoderLayer(*params)
        else:
            self.enc = nn.CellList([EncoderLayer(*params) for _ in range(num_layers)])
        
        self.layer_norm = nn.LayerNorm((hidden_size,))
        self.input_dropout = nn.Dropout(1 - input_dropout)

    def construct(self, inputs, mask):
        #Add input dropout
        x = self.input_dropout(inputs.astype(mindspore.float32))
        
        # Project to hidden size
        x = self.embedding_proj(x)
        
        if(self.universal):
            for l in range(self.num_layers):
                x += self.timing_signal[:, :inputs.shape[1], :].astype(inputs.dtype)
                x += mnp.tile(ops.expand_dims(self.position_signal[:, l, :], 1), (1,inputs.shape[1],1)).astype(inputs.dtype)
                x = self.enc(x, mask=mask)
            y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:, :inputs.shape[1], :].astype(inputs.dtype)
            
            for i in range(self.num_layers):
                x = self.enc[i](x, mask)
        
            y = self.layer_norm(x.astype(mindspore.float32))
        return y.astype(mindspore.float32)

class Decoder(nn.Cell):
    """
    A Transformer Decoder module. 
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """
    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=1000, input_dropout=0.0, layer_dropout=0.0, 
                 attention_dropout=0.0, relu_dropout=0.0, universal=False):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """
        
        super(Decoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)
        
        if(self.universal):  
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        self.mask = _get_attn_subsequent_mask(max_length)

        params =(hidden_size, 
                 total_key_depth or hidden_size,
                 total_value_depth or hidden_size,
                 filter_size, 
                 num_heads, 
                 _gen_bias_mask(max_length), # mandatory
                 layer_dropout, 
                 attention_dropout, 
                 relu_dropout)
        
        
        if(self.universal):
            self.dec = DecoderLayer(*params)
        else:
            self.dec = nn.SequentialCell(*[DecoderLayer(*params) for l in range(num_layers)])
        
        self.embedding_proj = nn.Dense(embedding_size, hidden_size, has_bias=False)
        self.layer_norm = nn.LayerNorm((hidden_size,))
        self.input_dropout = nn.Dropout(1 - input_dropout)

    def construct(self, inputs, encoder_output, mask):
        mask_src, mask_trg = mask
        dec_mask = ops.gt(mask_trg + self.mask[:, :mask_trg.shape[-1], :mask_trg.shape[-1]], 0)
        #Add input dropout
        x = self.input_dropout(inputs)
        if (not config.project): x = self.embedding_proj(x)
        
        if(self.universal):
            x += self.timing_signal[:, :inputs.shape[1], :].astype(inputs.dtype)
            for l in range(self.num_layers):
                x += mnp.tile(ops.expand_dims(self.position_signal[:, l, :], 1), (1, inputs.shape[1], 1)).astype(inputs.dtype)
                x, _, attn_dist, _ = self.dec((x, encoder_output, [], (mask_src, dec_mask)))
            y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:, :inputs.shape[1], :].astype(inputs.dtype)
            
            # Run decoder
            y, _, attn_dist, _ = self.dec((x, encoder_output, [], (mask_src,dec_mask)))

            # Final layer normalization
            y = self.layer_norm(y)
        return y.astype(mindspore.float32), attn_dist.astype(mindspore.float32)

class MulDecoder(nn.Cell):
    def __init__(self, expert_num,  embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=1000, input_dropout=0.0, layer_dropout=0.0, 
                 attention_dropout=0.0, relu_dropout=0.0):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """
        super(MulDecoder, self).__init__()
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)
        self.mask = _get_attn_subsequent_mask(max_length)

        params =(hidden_size, 
                 total_key_depth or hidden_size,
                 total_value_depth or hidden_size,
                 filter_size, 
                 num_heads, 
                 _gen_bias_mask(max_length), # mandatory
                 layer_dropout, 
                 attention_dropout, 
                 relu_dropout)
        if config.basic_learner: self.basic = DecoderLayer(*params)
        self.experts = nn.CellList([DecoderLayer(*params) for e in range(expert_num)])
        self.dec = nn.SequentialCell([DecoderLayer(*params) for l in range(num_layers)])
        
        self.embedding_proj = nn.Dense(embedding_size, hidden_size, has_bias=False)
        self.layer_norm = nn.LayerNorm((hidden_size,))
        self.input_dropout = nn.Dropout(1 - input_dropout)

    def construct(self, inputs, encoder_output, mask, attention_epxert):
        mask_src, mask_trg = mask
        dec_mask = ops.gt(mask_trg + self.mask[:, :mask_trg.shape[-1], :mask_trg.shape[-1]], 0)
        #Add input dropout
        x = self.input_dropout(inputs)
        x = self.embedding_proj(x.astype(mindspore.float32))
        # Add timing signal
        x += self.timing_signal[:, :inputs.shape[1], :].astype(inputs.dtype)
        expert_outputs = []
        if config.basic_learner:
            basic_out , _, attn_dist, _ = self.basic((x, encoder_output, [], (mask_src,dec_mask)))

        #compute experts
        #TODO forward all experts in parrallel
        if (attention_epxert.shape[0]==1 and config.topk > 0):
            for i, expert in enumerate(self.experts):
                if attention_epxert[0][i][0][0]>0.0001:         #speed up inference
                    expert_out , _, attn_dist, _ = expert((x, encoder_output, [], (mask_src,dec_mask)))
                    expert_outputs.append(attention_epxert[0][i][0][0]*expert_out)
            x = ops.stack(expert_outputs, axis=1)
            x = x.sum(axis=1)       
        else:
            for i, expert in enumerate(self.experts):
                expert_out , _, attn_dist, _ = expert((x, encoder_output, [], (mask_src,dec_mask)))
                expert_outputs.append(expert_out)
            x = ops.stack(expert_outputs, axis=1) #(batch_size, expert_number, len, hidden_size)
            x = attention_epxert * x
            x = x.sum(axis=1) #(batch_size, len, hidden_size)

        if config.basic_learner:
            x += basic_out
        # Run decoder
        y, _, attn_dist, _ = self.dec((x, encoder_output, [], (mask_src,dec_mask)))
        
        # Final layer normalization
        y = self.layer_norm(y)
        return y.astype(mindspore.float32), attn_dist.astype(mindspore.float32)

class Generator(nn.Cell):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Dense(d_model, vocab)

    def construct(self, x, attn_dist=None, temp=1, beam_search=False, attn_dist_db=None):
        logit = self.proj(x)

        log_softmax = nn.LogSoftmax(axis=-1)
        return log_softmax(logit).astype(mindspore.float32)

class Transformer_experts(nn.Cell):
    def __init__(self, vocab, decoder_number, model_file_path=None, is_eval=False, load_optim=False):
        super(Transformer_experts, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab.n_words

        self.embedding = share_embedding(self.vocab, config.pretrain_emb)
        self.encoder = Encoder(config.emb_dim, config.hidden_dim, num_layers=config.hop, num_heads=config.heads, 
                                total_key_depth=config.depth, total_value_depth=config.depth,
                                filter_size=config.filter, universal=config.universal)
        self.decoder_number = decoder_number
        ## multiple decoders
        self.decoder = MulDecoder(decoder_number, config.emb_dim, config.hidden_dim, num_layers=config.hop, num_heads=config.heads, 
                                    total_key_depth=config.depth, total_value_depth=config.depth,
                                    filter_size=config.filter)
 
        self.decoder_key = nn.Dense(config.hidden_dim, decoder_number, has_bias=False)

        self.generator = Generator(config.hidden_dim, self.vocab_size)
        
        self.attention_activation = nn.Softmax(axis=1)

        self.model_dir = config.save_path
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.best_path = ""

    def construct(self, inputs, iter, train=True):
        enc_batch = inputs["input_batch"].astype(mindspore.float32)
        dec_batch = inputs["target_batch"].astype(mindspore.float32)
        mask_input = inputs["mask_input"].astype(mindspore.float32)
        target_program = inputs["target_program"].astype(mindspore.float32)
        
        ## Encode
        mask_src = ops.expand_dims(ops.equal(enc_batch, config.PAD_idx), 1)
        emb_mask = self.embedding(mask_input).astype(mindspore.float32)
        encoder_outputs = self.encoder(self.embedding(enc_batch) + emb_mask, mask_src).astype(mindspore.float32)

        ## Attention over decoder
        q_h = encoder_outputs[:,0]
        logit_prob = self.decoder_key(q_h.astype(mindspore.float32)) #(bsz, num_experts)

        if(config.topk > 0):
            k_max_value, k_max_index = ops.top_k(logit_prob, config.topk)
            a = np.empty([logit_prob.shape[0], self.decoder_number])
            a.fill(float('-inf'))
            mask = Tensor(a)
            logit_prob_ = ops.tensor_scatter_elements(mask.astype(mindspore.float32), k_max_index.astype(mindspore.int32), k_max_value.astype(mindspore.float32), axis=1)
            attention_parameters = self.attention_activation(logit_prob_.astype(mindspore.float32))
        else:
            attention_parameters = self.attention_activation(logit_prob.astype(mindspore.float32))

        if(config.oracle): attention_parameters = self.attention_activation(Tensor(target_program).astype(mindspore.float32)*1000)
        
        attention_parameters = ops.expand_dims(ops.expand_dims(attention_parameters, -1), -1) # (batch_size, expert_num, 1, 1)

        # Decode 
        sos_token = ops.expand_dims(Tensor([config.SOS_idx] * enc_batch.shape[0]), 1)
        dec_batch_shift = ops.concat((sos_token.astype(mindspore.float32), dec_batch[:, :-1].astype(mindspore.float32)), axis=1)

        mask_trg = ops.expand_dims(ops.equal(dec_batch_shift, config.PAD_idx), 1)
        pre_logit, attn_dist = self.decoder(self.embedding(dec_batch_shift.astype(mindspore.float32)), encoder_outputs, (mask_src, mask_trg), attention_parameters)
        ## compute output dist
        logit = self.generator(pre_logit, attn_dist, None, None, attn_dist_db=None)
        
        ## loss: NNL if ptr else Cross entropy
        if(train and config.schedule>10):
            if(random.uniform(0, 1) <= (0.0001 + (1 - 0.0001) * math.exp(-1. * iter / config.schedule))):
                config.oracle=True
            else:
                config.oracle=False

        return logit.astype(mindspore.float32), logit_prob.astype(mindspore.float32)

class Loss(nn.LossBase):
    def __init__(self):
        super(Loss, self).__init__()
        self.criterion = nn.NLLLoss(ignore_index=config.PAD_idx)
        self.cross_entropy = nn.SoftmaxCrossEntropyWithLogits(sparse=True)

    def construct(self, logit, dec_batch, logit_prob, program_label):
        x = self.criterion(logit.view((-1, logit.shape[-1])), dec_batch.view(-1).astype(mindspore.int32)) + \
            self.cross_entropy(logit_prob, Tensor(program_label).astype(mindspore.int32))
        return self.get_loss(x)

