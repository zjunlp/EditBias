export CUDA_VISIBLE_DEVICES=1
python run.py \
    +alg=mend \
    +experiment=debias \
    +model=gpt2xl \
    data.wiki_webtext=False \
    batch_size=32 \
    val_batch_size=32 \
    eval_only=True \
    save_path=gpt2-xl-test_gender \
    archive=outputs/xxxx \
    val_set=data/stereoset/gender_test.json \
    lr=1e-6 \
    edit_lr=1e-5 \
    lr_lr=1e-6

python run.py \
    +alg=mend \
    +experiment=debias \
    +model=gpt2xl \
    data.wiki_webtext=False \
    batch_size=32 \
    val_batch_size=32 \
    eval_only=True \
    save_path=gpt2-xl-test_race \
    archive=outputs/xxxx \
    val_set=data/stereoset/race_test.json \
    lr=1e-6 \
    edit_lr=1e-5 \
    lr_lr=1e-6

python run.py \
    +alg=mend \
    +experiment=debias \
    +model=gpt2xl \
    data.wiki_webtext=False \
    batch_size=32 \
    val_batch_size=32 \
    eval_only=True \
    save_path=gpt2-xl-test_religion \
    archive=outputs/xxxx \
    val_set=data/stereoset/religion_test.json \
    lr=1e-6 \
    edit_lr=1e-5 \
    lr_lr=1e-6
