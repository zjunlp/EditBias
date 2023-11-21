export CUDA_VISIBLE_DEVICES=1
python -m run \
    +alg=mend \
    +experiment=debias \
    +model=gpt2 \
    data.wiki_webtext=False \
    batch_size=128 \
    val_batch_size=128 \
    save_path=edit_gpt2\
    lr=1e-5 \
    edit_lr=1e-4 \
    lr_lr=1e-5