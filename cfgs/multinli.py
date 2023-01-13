accelerator: gpu
batch_size: 16
check_val_every_n_epoch: 1
ckpt_every_n_epoch: 1
data_augmentation: False
datamodule: multinli
devices: 1
lr: 1e-5
lr_scheduler: linear
max_epochs: 10
model: bert
optimizer: adamw
refresh_rate: 12000
