# EditBias: Debiasing Language Models via Model Editing


## Setup

### Environment

This codebase uses Python 3.9.18. Other versions may work as well.

Create an environment
and install the dependencies:

    $ conda create -n editbias python=3.9
    $ conda activate editbias
    (editbias) $ pip install -r requirements.txt


## EditBias

- Training
    ```bash
     (editbias) $ bash scripts/gpt2-xl.sh >scripts/gpt2-xl.log 2>&1
    ```
    Record the path $p_1$ of the final parameters of the editor networks in the training log.

- Evaluation

    1. Set `archive` as $p_1$ in the evaluation script.

    2. Run `bash scripts/gpt2-xl_eval.sh >scripts/gpt2-xl_eval.log 2>&1` and record the output path $p_2$ in the evaluation log.

    3. Set `root` as $p_2$ in `res.py` and run `python res.py`

## Bias Tracing
Enter [bias_tracing](./bias_tracing)

Run the scripts `bash scripts/gpt2.sh` or `bash scripts/roberta.sh`.

## Citing the paper

If this code or paper was useful, please consider using the following citation:

    @article{xinxu2023EditBias,
        title={EditBias: Debiasing Language Models via Model Editing},
        author={Xin Xu, Wei Xu},
        year={2023},
        url={https://github.com/xxupiano/EditBias}
    }
