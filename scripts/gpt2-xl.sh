export CUDA_VISIBLE_DEVICES=0
python -m run \
    +alg=mend \
    +experiment=debias \
    +model=gpt2xl \
    data.wiki_webtext=False \
    batch_size=32 \
    val_batch_size=32 \
    save_path=gpt2-xl \
    lr=1e-6 \
    edit_lr=1e-5 \
    lr_lr=1e-6
