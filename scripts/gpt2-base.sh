export CUDA_VISIBLE_DEVICES=0
python -m run \
    +alg=mend \
    +experiment=debias \
    +model=gpt2_123 \
    dataset=stereoset \
    batch_size=64 \
    val_batch_size=64 \
    accumulate_bs=1 \
    lr=1e-7 \
    edit_lr=1e-8 \
    lr_lr=1e-7 \
    early_stop_patience=10

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
    early_stop_patience=10

python -m run \
    +alg=mend \
    +experiment=debias \
    +model=gpt2_1 \
    dataset=stereoset \
    batch_size=64 \
    val_batch_size=64 \
    accumulate_bs=1 \
    lr=1e-7 \
    edit_lr=1e-7 \
    lr_lr=1e-7 \
    early_stop_patience=10

python -m run \
    +alg=mend \
    +experiment=debias \
    +model=gpt2_2 \
    dataset=stereoset \
    batch_size=64 \
    val_batch_size=64 \
    accumulate_bs=1 \
    lr=1e-7 \
    edit_lr=1e-8 \
    lr_lr=1e-7 \
    early_stop_patience=10

python -m run \
    +alg=mend \
    +experiment=debias \
    +model=gpt2_3 \
    dataset=stereoset \
    batch_size=64 \
    val_batch_size=64 \
    accumulate_bs=1 \
    lr=1e-7 \
    edit_lr=1e-8 \
    lr_lr=1e-7 \
    early_stop_patience=10

python -m run \
    +alg=mend \
    +experiment=debias \
    +model=gpt2_last123 \
    dataset=stereoset \
    batch_size=128 \
    val_batch_size=128 \
    accumulate_bs=1 \
    lr=1e-7 \
    edit_lr=1e-8 \
    lr_lr=1e-7 \
    early_stop_patience=10

python -m run \
    +alg=mend \
    +experiment=debias \
    +model=gpt2_last12 \
    dataset=stereoset \
    batch_size=128 \
    val_batch_size=128 \
    accumulate_bs=1 \
    lr=1e-7 \
    edit_lr=1e-8 \
    lr_lr=1e-7 \
    early_stop_patience=10


python -m run \
    +alg=mend \
    +experiment=debias \
    +model=gpt2_last1 \
    dataset=stereoset \
    batch_size=64 \
    val_batch_size=64 \
    accumulate_bs=1 \
    lr=1e-7 \
    edit_lr=1e-8 \
    lr_lr=1e-7 \
    early_stop_patience=10

python -m run \
    +alg=mend \
    +experiment=debias \
    +model=gpt2_last2 \
    dataset=stereoset \
    batch_size=64 \
    val_batch_size=64 \
    accumulate_bs=1 \
    lr=1e-7 \
    edit_lr=1e-8 \
    lr_lr=1e-7 \
    early_stop_patience=10

python -m run \
    +alg=mend \
    +experiment=debias \
    +model=gpt2_last3 \
    dataset=stereoset \
    batch_size=64 \
    val_batch_size=64 \
    accumulate_bs=1 \
    lr=1e-7 \
    edit_lr=1e-8 \
    lr_lr=1e-7 \
    early_stop_patience=10
