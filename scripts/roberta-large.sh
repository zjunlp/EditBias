python -m run \
    +alg=mend \
    +experiment=debias \
    +model=roberta-large_12 \
    dataset=stereoset \
    batch_size=64 \
    val_batch_size=64 \
    accumulate_bs=1 \
    lr=1e-4 \
    edit_lr=1e-5 \
    lr_lr=1e-4 \
    early_stop_patience=15

python -m run \
    +alg=mend \
    +experiment=debias \
    +model=roberta-large_123 \
    dataset=stereoset \
    batch_size=128 \
    val_batch_size=128 \
    accumulate_bs=1 \
    lr=1e-8 \
    edit_lr=1e-9 \
    lr_lr=1e-8 \
    early_stop_patience=10

python -m run \
    +alg=mend \
    +experiment=debias \
    +model=roberta-large_1 \
    dataset=stereoset \
    batch_size=64 \
    val_batch_size=64 \
    accumulate_bs=1 \
    lr=1e-4 \
    edit_lr=1e-5 \
    lr_lr=1e-4 \
    early_stop_patience=15

python -m run \
    +alg=mend \
    +experiment=debias \
    +model=roberta-large_2 \
    dataset=stereoset \
    batch_size=64 \
    val_batch_size=64 \
    accumulate_bs=1 \
    lr=1e-4 \
    edit_lr=1e-5 \
    lr_lr=1e-4 \
    early_stop_patience=15

python -m run \
    +alg=mend \
    +experiment=debias \
    +model=roberta-large_3 \
    dataset=stereoset \
    batch_size=32 \
    val_batch_size=32 \
    accumulate_bs=1 \
    lr=1e-4 \
    edit_lr=1e-5 \
    lr_lr=1e-4 \
    early_stop_patience=15


python -m run \
    +alg=mend \
    +experiment=debias \
    +model=roberta-large_last123 \
    dataset=stereoset \
    batch_size=128 \
    val_batch_size=128 \
    accumulate_bs=1 \
    lr=1e-8 \
    edit_lr=1e-9 \
    lr_lr=1e-8 \
    early_stop_patience=10

# beng
python -m run \
    +alg=mend \
    +experiment=debias \
    +model=roberta-large_last12 \
    dataset=stereoset \
    batch_size=128 \
    val_batch_size=128 \
    accumulate_bs=1 \
    lr=1e-8 \
    edit_lr=1e-9 \
    lr_lr=1e-8 \
    early_stop_patience=10

python -m run \
    +alg=mend \
    +experiment=debias \
    +model=roberta-large_last3 \
    dataset=stereoset \
    batch_size=64 \
    val_batch_size=64 \
    accumulate_bs=1 \
    lr=1e-4 \
    edit_lr=1e-5 \
    lr_lr=1e-4 \
    early_stop_patience=15

python -m run \
    +alg=mend \
    +experiment=debias \
    +model=roberta-large_last2 \
    dataset=stereoset \
    batch_size=64 \
    val_batch_size=64 \
    accumulate_bs=1 \
    lr=1e-4 \
    edit_lr=1e-5 \
    lr_lr=1e-4 \
    early_stop_patience=15

python -m run \
    +alg=mend \
    +experiment=debias \
    +model=roberta-large_last1 \
    dataset=stereoset \
    batch_size=64 \
    val_batch_size=64 \
    accumulate_bs=1 \
    lr=1e-4 \
    edit_lr=1e-5 \
    lr_lr=1e-4 \
    early_stop_patience=15