import numpy as np

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops

from mindspore import context, log
from mindspore.common.initializer import initializer, XavierUniform

from model.transformer_mulexpert import Transformer_experts, Loss
from utils import config
from utils.data_loader import prepare_data_seq
from utils.beam_omt_experts import Translator
from utils.metric import moses_multi_bleu


if (config.USE_CUDA):
    mindspore.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
else:
    mindspore.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

mindspore.set_seed(0)
np.random.seed(0)

dataset_train, dataset_test, vocab, program_number = prepare_data_seq(batch_size=config.batch_size)

model = Transformer_experts(vocab, decoder_number=program_number)
for n, p in model.parameters_and_names():
    if p.ndim > 1 and (n != "embedding.lut.embedding_table" and config.pretrain_emb):
        p = initializer(XavierUniform(), p.shape, dtype=mindspore.float32)

# for n, p in model.parameters_and_names():
#     if p.dtype != mindspore.float32:
#         print(n, p)

loss_fn = Loss()
optimizer = nn.Adam(model.trainable_params(), learning_rate=config.lr)

def train(model, dataset, loss_fn, optimizer):
    def forward_fn(inputs):
        input_batch = inputs["input_batch"].astype(mindspore.float32)
        target_batch = inputs["target_batch"].astype(mindspore.float32)
        mask_input = inputs["mask_input"].astype(mindspore.float32)
        target_program = inputs["target_program"].astype(mindspore.float32)
        
        logit, logit_prob = model(input_batch, target_batch, mask_input, target_program)

        program_label = inputs["program_label"].astype(mindspore.float32)
        loss = loss_fn(logit, target_batch, logit_prob, program_label)
        return loss

    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=False)

    def train_step(inputs):
        loss, grads = grad_fn(inputs)
        loss = ops.depend(loss, optimizer(grads))
        return loss

    size = dataset.get_dataset_size()
    model.set_train()
    for batch, inputs in enumerate(dataset.create_dict_iterator()):
        # Compute prediction error
        loss = train_step(inputs)

        if batch % 100 == 0:
            loss, current = loss.asnumpy(), batch
            print(f"loss: {loss:>7f}  [{current:>3d}/{size:>3d}]")

def test(model, dataset, loss_fn):
    num_batches = dataset.get_dataset_size()
    model.set_train(False)

    ref, hyp_b = [], []
    t = Translator(model, model.vocab)

    test_loss= 0
    for batch, inputs in enumerate(dataset.create_dict_iterator()):
        sent_b = t.beam_search(inputs, max_dec_step=50)
        for i, beam_sent in enumerate(sent_b):
            target_batch = list(inputs["target_batch"][i].asnumpy())
            target_txt = []
            for index in target_batch:
                index = index.tolist()
                if index in model.vocab.index2word and index not in [0, 1, 2, 3, 4, 5, 6]:
                    target_txt.append(model.vocab.index2word[index])
                else:
                    continue
            rf = " ".join(target_txt)
            hyp_b.append(beam_sent)
            ref.append(rf)
            input_batch = list(inputs["input_batch"][i].asnumpy())
            input_txt = []
            for index in input_batch:
                index = index.tolist()
                if index in model.vocab.index2word and index not in [0, 1, 2, 3, 4, 5, 6]:
                    input_txt.append(model.vocab.index2word[index])
                else:
                    continue
            print_custum(dial=[" ".join(input_txt)], ref=rf, hyp_b=beam_sent)   

        bleu_score_b = moses_multi_bleu(np.array(hyp_b), np.array(ref), lowercase=True)

        logit, logit_prob = model(inputs, batch)

        target_batch = inputs["target_batch"]
        program_label = inputs["program_label"]
        test_loss += loss_fn(logit, target_batch, logit_prob, program_label).asnumpy()

    test_loss /= num_batches
    # print(f"Test Error: \n BLEU score: {bleu_score}, Avg loss: {test_loss:>8f} \n")
    print(f"Test: \n BLEU score: {bleu_score_b:>0.1f}, Avg loss: {test_loss:>8f} \n")

def print_custum(dial, ref, hyp_b):
    print("Context:{}".format(dial))
    print("Beam: {}".format(hyp_b))
    print("Ref:{}".format(ref))
    print("----------------------------------------------------------------------")
    print("----------------------------------------------------------------------")

epochs = 1
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(model, dataset_train, loss_fn, optimizer)
    test(model, dataset_test, loss_fn)
print("Done!")