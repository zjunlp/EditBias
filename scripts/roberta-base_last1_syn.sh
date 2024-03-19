export CUDA_VISIBLE_DEVICES=1
python -m run \
    +alg=mend \
    +experiment=debias \
    +model=roberta-base_last1 \
    dataset=stereoset \
    batch_size=32 \
    val_batch_size=32 \
    eval_only=True \
    archive=outputs/xxxx.bk \
    val_set=data/stereoset/gender_test_syn.json

python -m run \
    +alg=mend \
    +experiment=debias \
    +model=roberta-base_last1 \
    dataset=stereoset \
    batch_size=32 \
    val_batch_size=32 \
    eval_only=True \
    archive=outputs/xxxx.bk \
    val_set=data/stereoset/race_test_syn.json

python -m run \
    +alg=mend \
    +experiment=debias \
    +model=roberta-base_last1 \
    dataset=stereoset \
    batch_size=32 \
    val_batch_size=32 \
    eval_only=True \
    archive=outputs/xxxx.bk \
    val_set=data/stereoset/religion_test_syn.json