export CUDA_VISIBLE_DEVICES=1
python -m run \
    +alg=mend \
    +experiment=debias \
    +model=gpt2m_123 \
    dataset=stereoset \
    batch_size=90 \
    val_batch_size=90 \
    accumulate_bs=1 \
    lr=1e-7 \
    edit_lr=1e-8 \
    lr_lr=1e-7 \
    early_stop_patience=10

python -m run \
    +alg=mend \
    +experiment=debias \
    +model=gpt2m_12 \
    dataset=stereoset \
    batch_size=90 \
    val_batch_size=90 \
    accumulate_bs=1 \
    lr=1e-7 \
    edit_lr=1e-8 \
    lr_lr=1e-7 \
    early_stop_patience=10

python -m run \
    +alg=mend \
    +experiment=debias \
    +model=gpt2m_1 \
    dataset=stereoset \
    batch_size=90 \
    val_batch_size=90 \
    accumulate_bs=1 \
    lr=1e-7 \
    edit_lr=1e-8 \
    lr_lr=1e-7 \
    early_stop_patience=10

python -m run \
    +alg=mend \
    +experiment=debias \
    +model=gpt2m_2 \
    dataset=stereoset \
    batch_size=90 \
    val_batch_size=90 \
    accumulate_bs=1 \
    lr=1e-7 \
    edit_lr=1e-8 \
    lr_lr=1e-7 \
    early_stop_patience=10

python -m run \
    +alg=mend \
    +experiment=debias \
    +model=gpt2m_3 \
    dataset=stereoset \
    batch_size=90 \
    val_batch_size=90 \
    accumulate_bs=1 \
    lr=1e-7 \
    edit_lr=1e-8 \
    lr_lr=1e-7 \
    early_stop_patience=10

python -m run \
    +alg=mend \
    +experiment=debias \
    +model=gpt2m_last123 \
    dataset=stereoset \
    batch_size=90 \
    val_batch_size=90 \
    accumulate_bs=1 \
    lr=1e-7 \
    edit_lr=1e-8 \
    lr_lr=1e-7 \
    early_stop_patience=10

python -m run \
    +alg=mend \
    +experiment=debias \
    +model=gpt2m_last12 \
    dataset=stereoset \
    batch_size=90 \
    val_batch_size=90 \
    accumulate_bs=1 \
    lr=1e-7 \
    edit_lr=1e-8 \
    lr_lr=1e-7 \
    early_stop_patience=10

python -m run \
    +alg=mend \
    +experiment=debias \
    +model=gpt2m_last1 \
    dataset=stereoset \
    batch_size=90 \
    val_batch_size=90 \
    accumulate_bs=1 \
    lr=1e-7 \
    edit_lr=1e-8 \
    lr_lr=1e-7 \
    early_stop_patience=10

python -m run \
    +alg=mend \
    +experiment=debias \
    +model=gpt2m_last2 \
    dataset=stereoset \
    batch_size=90 \
    val_batch_size=90 \
    accumulate_bs=1 \
    lr=1e-7 \
    edit_lr=1e-8 \
    lr_lr=1e-7 \
    early_stop_patience=10

python -m run \
    +alg=mend \
    +experiment=debias \
    +model=gpt2m_last3 \
    dataset=stereoset \
    batch_size=90 \
    val_batch_size=90 \
    accumulate_bs=1 \
    lr=1e-7 \
    edit_lr=1e-8 \
    lr_lr=1e-7 \
    early_stop_patience=10