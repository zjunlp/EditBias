export CUDA_VISIBLE_DEVICES=0
python -m run \
    +alg=mend \
    +experiment=debias \
    +model=gpt2_123 \
    dataset=stereoset \
    batch_size=64 \
    val_batch_size=64 \
    eval_only=True \
    archive=outputs/xxxx.bk \
    val_set=data/stereoset/gender_test.json

python -m run \
    +alg=mend \
    +experiment=debias \
    +model=gpt2_123 \
    dataset=stereoset \
    batch_size=64 \
    val_batch_size=64 \
    eval_only=True \
    archive=outputs/xxxx.bk \
    val_set=data/stereoset/race_test.json

python -m run \
    +alg=mend \
    +experiment=debias \
    +model=gpt2_123 \
    dataset=stereoset \
    batch_size=64 \
    val_batch_size=64 \
    eval_only=True \
    archive=outputs/xxxx.bk \
    val_set=data/stereoset/religion_test.json
