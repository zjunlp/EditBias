export CUDA_VISIBLE_DEVICES=0
python run.py \
    +alg=mend \
    +experiment=debias \
    +model=gpt2 \
    data.wiki_webtext=False \
    batch_size=128 \
    val_batch_size=128 \
    eval_only=True \
    save_path=gpt2-test_gender \
    archive=outputs/xxxx \
    val_set=data/stereoset/gender_test.json \
    lr=1e-5 \
    edit_lr=1e-4 \
    lr_lr=1e-5

python run.py \
    +alg=mend \
    +experiment=debias \
    +model=gpt2 \
    data.wiki_webtext=False \
    batch_size=128 \
    val_batch_size=128 \
    eval_only=True \
    save_path=gpt2-test_race \
    archive=outputs/xxxx \
    val_set=data/stereoset/race_test.json \
    lr=1e-5 \
    edit_lr=1e-4 \
    lr_lr=1e-5

python run.py \
    +alg=mend \
    +experiment=debias \
    +model=gpt2 \
    data.wiki_webtext=False \
    batch_size=128 \
    val_batch_size=128 \
    eval_only=True \
    save_path=gpt2-test_religion \
    archive=outputs/xxxx \
    val_set=data/stereoset/religion_test.json \
    lr=1e-5 \
    edit_lr=1e-4 \
    lr_lr=1e-5