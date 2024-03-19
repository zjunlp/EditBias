python -m run \
    +alg=mend \
    +experiment=debias \
    +model=roberta-base_12 \
    dataset=stereoset \
    batch_size=64 \
    val_batch_size=64 \
    accumulate_bs=1 \
    cloc=1.5 \
    lr=1e-6 \
    edit_lr=1e-7 \
    lr_lr=1e-6 \
    early_stop_patience=7

python -m run \
    +alg=mend \
    +experiment=debias \
    +model=roberta-base_123 \
    dataset=stereoset \
    batch_size=64 \
    val_batch_size=64 \
    accumulate_bs=1 \
    cloc=1.5 \
    lr=1e-6 \
    edit_lr=1e-7 \
    lr_lr=1e-6 \
    early_stop_patience=7

python -m run \
    +alg=mend \
    +experiment=debias \
    +model=roberta-base_1 \
    dataset=stereoset \
    batch_size=64 \
    val_batch_size=64 \
    accumulate_bs=1 \
    lr=1e-6 \
    edit_lr=1e-7 \
    lr_lr=1e-6 \
    early_stop_patience=7


python -m run \
    +alg=mend \
    +experiment=debias \
    +model=roberta-base_2 \
    dataset=stereoset \
    batch_size=64 \
    val_batch_size=64 \
    accumulate_bs=1 \
    lr=1e-4 \
    edit_lr=1e-5 \
    lr_lr=1e-4 \
    early_stop_patience=10


python -m run \
    +alg=mend \
    +experiment=debias \
    +model=roberta-base_3 \
    dataset=stereoset \
    batch_size=54 \
    val_batch_size=54 \
    accumulate_bs=1 \
    cloc=1.5 \
    lr=1e-7 \
    edit_lr=1e-8 \
    lr_lr=1e-7 \
    early_stop_patience=10


python -m run \
    +alg=mend \
    +experiment=debias \
    +model=roberta-base_last3 \
    dataset=stereoset \
    batch_size=64 \
    val_batch_size=64 \
    accumulate_bs=1 \
    cloc=1.5 \
    lr=1e-6 \
    edit_lr=1e-7 \
    lr_lr=1e-6 \
    early_stop_patience=10

python -m run \
    +alg=mend \
    +experiment=debias \
    +model=roberta-base_last2 \
    dataset=stereoset \
    batch_size=64 \
    val_batch_size=64 \
    accumulate_bs=1 \
    cloc=1.5 \
    lr=1e-6 \
    edit_lr=1e-7 \
    lr_lr=1e-6 \
    early_stop_patience=10

python -m run \
    +alg=mend \
    +experiment=debias \
    +model=roberta-base_last1 \
    dataset=stereoset \
    batch_size=64 \
    val_batch_size=64 \
    accumulate_bs=1 \
    cloc=1.5 \
    lr=1e-6 \
    edit_lr=1e-7 \
    lr_lr=1e-6 \
    early_stop_patience=10

python -m run \
    +alg=mend \
    +experiment=debias \
    +model=roberta-base_last21 \
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
    +model=roberta-base_last321 \
    dataset=stereoset \
    batch_size=64 \
    val_batch_size=64 \
    accumulate_bs=1 \
    lr=1e-7 \
    edit_lr=1e-8 \
    lr_lr=1e-7 \
    early_stop_patience=10