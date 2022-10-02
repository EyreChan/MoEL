import os
import pickle

import mindspore
import mindspore.dataset as ds

from mindspore import log, Tensor

from utils import config
from utils.data_reader import load_dataset
# from model.common_layer import write_config

mindspore.set_seed(0)

class Dataset():
    def __init__(self, data, vocab):
        """Reads source and target sequences from txt files."""
        self.vocab = vocab
        self.data = data 
        self.emo_map = {
        'surprised': 0, 'excited': 1, 'annoyed': 2, 'proud': 3, 'angry': 4, 'sad': 5, 'grateful': 6, 'lonely': 7,
        'impressed': 8, 'afraid': 9, 'disgusted': 10, 'confident': 11, 'terrified': 12, 'hopeful': 13, 'anxious': 14, 'disappointed': 15,
        'joyful': 16, 'prepared': 17, 'guilty': 18, 'furious': 19, 'nostalgic': 20, 'jealous': 21, 'anticipating': 22, 'embarrassed': 23,
        'content': 24, 'devastated': 25, 'sentimental': 26, 'caring': 27, 'trusting': 28, 'ashamed': 29, 'apprehensive': 30, 'faithful': 31}
    
    def __len__(self):
        return len(self.data["target"])

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        self.context_text = self.data["context"][index]
        self.target_text = self.data["target"][index]
        self.emotion_text = self.data["emotion"][index]

        self.context, self.context_len, self.context_mask, self.context_mask_len = self.preprocess(self.context_text)

        self.target, self.target_len = self.preprocess(self.target_text, anw=True)
        self.emotion, self.emotion_label = self.preprocess_emo(self.emotion_text, self.emo_map)

        return self.context, self.context_len, self.context_mask, self.target, self.target_len, \
                self.emotion, self.emotion_label

    def preprocess(self, arr, anw=False):
        """Converts words to ids."""
        if(anw):
            sequence = [self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_idx for word in arr] + [config.EOS_idx]
            length = len(sequence)
            if length < 120:
                sequence += [config.PAD_idx] * (120 - length)
            return Tensor(sequence), length
        else:
            X_dial = [config.CLS_idx]
            X_mask = [config.CLS_idx]
            for i, sentence in enumerate(arr):
                X_dial += [self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_idx for word in sentence]
                spk = self.vocab.word2index["USR"] if i % 2 == 0 else self.vocab.word2index["SYS"]
                X_mask += [spk for _ in range(len(sentence))]

            assert len(X_dial) == len(X_mask)
            length_X_dial = len(X_dial)
            length_X_mask = len(X_mask)
            if length_X_dial < 400:
                X_dial += [config.PAD_idx] * (400 - length_X_dial)
                X_mask += [config.PAD_idx] * (400 - length_X_mask)
            
            return Tensor(X_dial), length_X_dial, Tensor(X_mask), length_X_mask

    def preprocess_emo(self, emotion, emo_map):
        program = [0]*len(emo_map)
        program[emo_map[emotion]] = 1
        return program, emo_map[emotion]

def prepare_data_seq(batch_size=32):  
    pairs_tra, pairs_tst, vocab = load_dataset()

    log.warning("Vocab  {} ".format(vocab.n_words))

    dataset_train = Dataset(pairs_tra, vocab)
    emo_map = dataset_train.emo_map
    dataset_train = ds.GeneratorDataset(dataset_train, ["input_batch", "input_lengths", \
                    "mask_input", "target_batch", "target_lengths", "target_program", \
                    "program_label"], shuffle=True)
    dataset_train = dataset_train.batch(batch_size=batch_size)
    # iterator = dataset_train.create_dict_iterator()
    # log.warning(next(iter(iterator)))

    dataset_test = Dataset(pairs_tst, vocab)
    dataset_test = ds.GeneratorDataset(dataset_test, ["input_batch", "input_lengths", \
                    "mask_input", "target_batch", "target_lengths", "target_program", \
                    "program_label"], shuffle=True)
    dataset_test = dataset_test.batch(batch_size=1)
    # iterator = dataset_test.create_dict_iterator()
    # log.warning(next(iter(iterator)))

    # write_config()
    return dataset_train, dataset_test, vocab, len(emo_map)

def prepare_data_seq_without_bz():  
    pairs_tra, pairs_tst, vocab = load_dataset()

    log.warning("Vocab  {} ".format(vocab.n_words))

    dataset_train = Dataset(pairs_tra, vocab)
    emo_map = dataset_train.emo_map
    dataset_train = ds.GeneratorDataset(dataset_train, ["input_batch", "input_lengths", \
                    "mask_input", "target_batch", "target_lengths", "target_program", \
                    "program_label"], shuffle=True)

    dataset_test = Dataset(pairs_tst, vocab)
    dataset_test = ds.GeneratorDataset(dataset_test, ["input_batch", "input_lengths", \
                    "mask_input", "target_batch", "target_lengths", "target_program", \
                    "program_label"], shuffle=True)

    # write_config()
    return dataset_train, dataset_test, vocab, len(emo_map)