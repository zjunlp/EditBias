export CUDA_VISIBLE_DEVICES=0
python -m run \
    +alg=mend \
    +experiment=debias \
    +model=gpt2_12 \
    dataset=stereoset \
    batch_size=64 \
    val_batch_size=64 \
    eval_only=True \
    archive=outputs/xxxx.bk \
    val_set=data/stereoset/gender_test_syn.json

python -m run \
    +alg=mend \
    +experiment=debias \
    +model=gpt2_12 \
    dataset=stereoset \
    batch_size=64 \
    val_batch_size=64 \
    eval_only=True \
    archive=outputs/xxxx.bk \
    val_set=data/stereoset/race_test_syn.json

python -m run \
    +alg=mend \
    +experiment=debias \
    +model=gpt2_12 \
    dataset=stereoset \
    batch_size=64 \
    val_batch_size=64 \
    eval_only=True \
    archive=outputs/xxxx.bk \
    val_set=data/stereoset/religion_test_syn.json