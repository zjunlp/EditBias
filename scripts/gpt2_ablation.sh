export CUDA_VISIBLE_DEVICES=0
python -m run \
    +alg=mend \
    +experiment=debias \
    +model=gpt2_12 \
    dataset=stereoset \
    batch_size=64 \
    val_batch_size=64 \
    accumulate_bs=1 \
    lr=1e-7 \
    edit_lr=1e-8 \
    lr_lr=1e-7 \
    early_stop_patience=10 \
    ifloc=False
