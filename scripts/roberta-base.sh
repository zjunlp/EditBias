export CUDA_VISIBLE_DEVICES=0
python -m run \
    +alg=mend \
    +experiment=debias \
    +model=roberta-base \
    data.wiki_webtext=False \
    eval_only=False \
    batch_size=32 \
    val_batch_size=32 \
    save_path=edited_roberta-base \
    lr=1e-4 \
    edit_lr=1e-5 \
    lr_lr=1e-4 \
