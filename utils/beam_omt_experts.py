import numpy as np

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp

from mindspore import log, Tensor

from utils import config

class Beam():
    ''' Beam search '''
    def __init__(self, size, device=False):
        self.size = size
        self._done = False

        # The score for each translation on the beam.
        self.scores = mnp.full((size, ), 0, dtype=mindspore.float32)
        self.all_scores = []

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        self.next_ys = [mnp.full((size, ), config.PAD_idx, dtype=mindspore.int32)]
        self.next_ys[0][0] = config.SOS_idx

    def get_current_state(self):
        "Get the outputs for the current timestep."
        return self.get_tentative_hypothesis()

    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]

    @property
    def done(self):
        return self._done

    def advance(self, word_prob):
        "Update beam status and check if finished or not."
        num_words = word_prob.shape[1]

        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            beam_lk = word_prob + ops.expand_dims(self.scores, 1).expand_as(word_prob)
        else:
            beam_lk = word_prob[0]

        flat_beam_lk = beam_lk.view(-1)

        best_scores, best_scores_id = flat_beam_lk.top_k(self.size, True) # 1st sort
        best_scores, best_scores_id = flat_beam_lk.top_k(self.size, True) # 2nd sort

        self.all_scores.append(self.scores)
        self.scores = best_scores

        # bestScoresId is flattened as a (beam x word) array,
        # so we need to calculate which word and beam each score came from
        prev_k = best_scores_id // num_words
        self.prev_ks.append(prev_k)
        self.next_ys.append(best_scores_id - prev_k * num_words)

        # End condition is when top-of-beam is EOS.
        if self.next_ys[-1][0].asnumpy().tolist() == config.EOS_idx:
            self._done = True
            self.all_scores.append(self.scores)

        return self._done

    def sort_scores(self):
        "Sort the scores."
        sort = ops.Sort(0, True)
        return sort(self.scores)

    def get_the_best_score_and_idx(self):
        "Get the score of the best in the beam."
        scores, ids = self.sort_scores()
        return scores[1], ids[1]

    def get_tentative_hypothesis(self):
        "Get the decoded sequence for the current timestep."
        if len(self.next_ys) == 1:
            dec_seq = ops.expand_dims(self.next_ys[0], 1)
        else:
            _, keys = self.sort_scores()
            hyps = [self.get_hypothesis(k) for k in keys]
            hyps = [[config.SOS_idx] + h for h in hyps]
            dec_seq = Tensor(hyps).astype(mindspore.int32)

        return dec_seq

    def get_hypothesis(self, k):
        """ Walk back to construct the full hypothesis. """
        hyp = []
        for j in range(len(self.prev_ks) - 1, -1, -1):
            hyp.append(self.next_ys[j+1][k])
            k = self.prev_ks[j][k]

        return list(map(lambda x: x.asnumpy().tolist(), hyp[::-1]))

class Translator(object):
    ''' Load with trained model and handle the beam search '''
    def __init__(self, model, lang):
        self.model = model
        self.lang = lang
        self.vocab_size = lang.n_words
        self.beam_size = config.beam_size

    def beam_search(self, src_seq, max_dec_step):
        ''' Translation work in one batch '''

        def get_inst_idx_to_tensor_position_map(inst_idx_list):
            ''' Indicate the position of an instance in a tensor. '''
            return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

        def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
            ''' Collect tensor parts associated to active instances. '''

            _, *d_hs = beamed_tensor.shape
            n_curr_active_inst = len(curr_active_inst_idx)
            new_shape = (n_curr_active_inst * n_bm, *d_hs)

            beamed_tensor = beamed_tensor.view((n_prev_active_inst, -1))
            beamed_tensor = beamed_tensor.gather(curr_active_inst_idx, 0)
            beamed_tensor = beamed_tensor.view((*new_shape))

            return beamed_tensor

        def collate_active_info(src_seq, encoder_db, src_enc, inst_idx_to_position_map, active_inst_idx_list):
            # Sentences which are still active are collected,
            # so the decoder will not run on completed sentences.
            n_prev_active_inst = len(inst_idx_to_position_map)
            active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
            active_inst_idx = Tensor(active_inst_idx).astype(mindspore.int32)

            active_src_seq = collect_active_part(src_seq, active_inst_idx, n_prev_active_inst, n_bm)
            active_src_enc = collect_active_part(src_enc, active_inst_idx, n_prev_active_inst, n_bm)
            
            active_encoder_db = None

            active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            return active_src_seq, active_encoder_db, active_src_enc, active_inst_idx_to_position_map

        def beam_decode_step(inst_dec_beams, len_dec_seq, src_seq, enc_output, inst_idx_to_position_map, n_bm, mask_src, encoder_db, mask_transformer_db, DB_ext_vocab_batch):
            ''' Decode and update beam status, and then return active beam idx '''
            def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
                dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
                dec_partial_seq = ops.stack(dec_partial_seq)
                dec_partial_seq = dec_partial_seq.view((-1, len_dec_seq))
                return dec_partial_seq

            def prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm):
                dec_partial_pos = mnp.arange(1, len_dec_seq + 1, dtype=mindspore.int32)
                dec_partial_pos = mnp.tile(ops.expand_dims(dec_partial_pos, 0), (n_active_inst * n_bm, 1))
                return dec_partial_pos
                  
            def predict_word(dec_seq, dec_pos, src_seq, enc_output, n_active_inst, n_bm, mask_src, encoder_db, mask_transformer_db, atten):
                ## masking
                mask_trg = ops.expand_dims(ops.equal(dec_seq, config.PAD_idx), 1)
                mask_src = ops.concat([ops.expand_dims(mask_src[0], 0)]*mask_trg.shape[0], axis=0)

                dec_output, attn_dist = self.model.decoder(self.model.embedding(dec_seq), enc_output, (mask_src, mask_trg), atten)

                db_dist = None

                prob = self.model.generator(dec_output, attn_dist, 1, True, attn_dist_db=db_dist)
                #prob = F.log_softmax(prob,dim=-1) #fix the name later
                word_prob = prob[:, -1]
                word_prob = word_prob.view((n_active_inst, n_bm, -1))
                return word_prob

            def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map):
                active_inst_idx_list = []
                for inst_idx, inst_position in inst_idx_to_position_map.items():
                    is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])
                    if not is_inst_complete:
                        active_inst_idx_list += [inst_idx]
                return active_inst_idx_list

            n_active_inst = len(inst_idx_to_position_map)
            dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
            dec_pos = prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm)
            word_prob = predict_word(dec_seq, dec_pos, src_seq, enc_output, n_active_inst, n_bm, mask_src, encoder_db, mask_transformer_db, atten = self.attention_parameters)

            # Update the beam with predicted word prob information and collect incomplete instances
            active_inst_idx_list = collect_active_inst_idx_list(inst_dec_beams, word_prob, inst_idx_to_position_map)

            return active_inst_idx_list

        def collect_hypothesis_and_scores(inst_dec_beams, n_best):
            all_hyp, all_scores = [], []
            for inst_idx in range(len(inst_dec_beams)):
                scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
                all_scores += [scores[:n_best]]

                hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
                all_hyp += [hyps]
            return all_hyp, all_scores

        #-- Encode
        enc_batch = src_seq["input_batch"]

        mask_src = ops.expand_dims(ops.equal(enc_batch, config.PAD_idx), 1)
        emb_mask = self.model.embedding(src_seq["mask_input"])
        src_enc = self.model.encoder(self.model.embedding(enc_batch) + emb_mask, mask_src)
        encoder_db = None
        mask_transformer_db = None
        DB_ext_vocab_batch = None

        ## Attention over decoder
        q_h = src_enc[:,0]
        logit_prob = self.model.decoder_key(q_h)

        if(config.topk>0): 
            k_max_value, k_max_index = ops.top_k(logit_prob, config.topk)
            a = np.empty([logit_prob.shape[0], self.model.decoder_number])
            a.fill(float('-inf'))
            mask = Tensor(a)
            logit_prob = ops.tensor_scatter_elements(mask.astype(mindspore.float32), k_max_index, k_max_value, axis=1)

        attention_parameters = self.model.attention_activation(logit_prob)

        if(config.oracle): attention_parameters = self.model.attention_activation(Tensor(src_seq['target_program'], dtype=mindspore.float32)*1000)
        self.attention_parameters = ops.expand_dims(ops.expand_dims(attention_parameters, -1), -1)

        #-- Repeat data for beam search
        n_bm = self.beam_size
        n_inst, len_s, d_h = src_enc.shape
        _, self.len_program, _, _ = self.attention_parameters.shape
        src_seq = mnp.tile(enc_batch, (1, n_bm)).view((n_inst * n_bm, len_s))
        src_enc = mnp.tile(src_enc, (1, n_bm, 1)).view((n_inst * n_bm, len_s, d_h))
        #-- Prepare beams
        inst_dec_beams = [Beam(n_bm) for _ in range(n_inst)]

        #-- Bookkeeping for active or not
        active_inst_idx_list = list(range(n_inst))
        inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

        #-- Decode
        for len_dec_seq in range(1, max_dec_step + 1):
            active_inst_idx_list = beam_decode_step(inst_dec_beams, len_dec_seq, src_seq, src_enc, inst_idx_to_position_map, n_bm, mask_src, encoder_db, mask_transformer_db, DB_ext_vocab_batch)

            if not active_inst_idx_list:
                break  # all instances have finished their path to <EOS>

            src_seq, encoder_db, src_enc, inst_idx_to_position_map = collate_active_info(src_seq, encoder_db, src_enc, inst_idx_to_position_map, active_inst_idx_list)

        batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, 1)

        ret_sentences = []
        for d in batch_hyp:
            ret_sentences.append(' '.join([self.model.vocab.index2word[idx] for idx in d [0]]).replace('EOS',''))
                
        return ret_sentences#, batch_scores