export CUDA_VISIBLE_DEVICES=0
python -m run \
    +alg=mend \
    +experiment=debias \
    +model=gpt2m_12 \
    dataset=stereoset \
    batch_size=90 \
    val_batch_size=90 \
    eval_only=True \
    archive=outputs/xxx.bk \
    val_set=data/stereoset/gender_test_reverse.json

