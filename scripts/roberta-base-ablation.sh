export CUDA_VISIBLE_DEVICES=0
python -m run \
    +alg=mend \
    +experiment=debias \
    +model=roberta-base_12 \
    dataset=stereoset \
    batch_size=64 \
    val_batch_size=64 \
    accumulate_bs=1 \
    lr=1e-6 \
    edit_lr=1e-7 \
    lr_lr=1e-6 \
    early_stop_patience=7 \
    ifloc=False