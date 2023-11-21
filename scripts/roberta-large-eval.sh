export CUDA_VISIBLE_DEVICES=0
python run2.py \
    +alg=mend \
    +experiment=debias \
    +model=roberta-large \
    data.wiki_webtext=False \
    batch_size=32 \
    val_batch_size=32 \
    eval_only=True \
    save_path=/mnt/16t/xxu/mend-bias/roberta-large_gender \
    archive=/mnt/16t/xxu/mend-bias/outputs/2023-11-17_14-36-50_4527467709/models/roberta-large.2023-11-17_14-36-50_4527467709_250 \
    val_set=/mnt/16t/xxu/mend-bias/data/stereoset/gender_test.json \
    lr=1e-6 \
    edit_lr=1e-5 \
    lr_lr=1e-6

python run2.py \
    +alg=mend \
    +experiment=debias \
    +model=roberta-large \
    data.wiki_webtext=False \
    batch_size=32 \
    val_batch_size=32 \
    eval_only=True \
    save_path=/mnt/16t/xxu/mend-bias/roberta-large_race \
    archive=/mnt/16t/xxu/mend-bias/outputs/2023-11-17_14-36-50_4527467709/models/roberta-large.2023-11-17_14-36-50_4527467709_250 \
    val_set=/mnt/16t/xxu/mend-bias/data/stereoset/race_test.json \
    lr=1e-6 \
    edit_lr=1e-5 \
    lr_lr=1e-6