import mindspore.nn as nn

from mindnlp.engine.trainer import Trainer
from mindnlp.common.metrics import BleuScore
from model.transformer_mulexpert import Transformer_experts, Loss2
from utils.data_loader import prepare_data_seq_without_bz

# load datasets
dataset_train, dataset_test, vocab, program_number = prepare_data_seq_without_bz()

# define Models & Loss & Optimizer
lr = 0.0001

model = Transformer_experts(vocab, decoder_number=program_number)
loss_fn = Loss2()
optimizer = nn.Adam(model.trainable_params(), learning_rate=lr)

# define metrics
metric = BleuScore()

# define trainer
trainer = Trainer(network=model, train_dataset=dataset_train, eval_dataset=dataset_test, metrics=metric,
                  epochs=1, batch_size=1, loss_fn=loss_fn, optimizer=optimizer)
trainer.run(mode="pynative", tgt_columns=["target_batch", "program_label"])
print("end train")