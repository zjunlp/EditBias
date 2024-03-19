import copy
import random
import importlib
import logging
import hydra
from omegaconf import OmegaConf
import numpy as np
import torch
import utils

from trainer import EditTrainer
import models

from data_classes.stereoset import StereoSetDataset

OmegaConf.register_new_resolver("uuid", lambda: utils.uuid())


logging.basicConfig(format='%(asctime)s - %(levelname)s [%(filename)s:%(lineno)d] %(message)s',
                    level=logging.INFO)
LOG = logging.getLogger(__name__)




@hydra.main(config_path='config', config_name='config')
def run(config):
    LOG.info(f"\n\n{OmegaConf.to_yaml(config)}\n")
    base_dir = hydra.utils.get_original_cwd()
    LOG.info(f"Project base directory: {base_dir}")

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    model = models.get_model(config)
    tokenizer = models.get_tokenizer(config)

    model_name = model.__class__.__name__.lower()
    if "gpt" in model_name or "llama" in model_name:
        tokenizer.pad_token = tokenizer.eos_token

    if not config.eval_only:
        train_set = StereoSetDataset(tokenizer, f"{base_dir}/data/stereoset/train.json", config, model_name)
        val_set = StereoSetDataset(tokenizer, f"{base_dir}/data/stereoset/dev.json", config, model_name)
    else:
        train_set = StereoSetDataset(tokenizer, f"{base_dir}/data/stereoset/train.json", config, model_name)
        val_set = StereoSetDataset(tokenizer, config.val_set, config, model_name)

    alg_module = importlib.import_module(f"algs.{config.alg}")
    LOG.info(f"Loading class {config.alg.upper()} from module {alg_module}")
    AlgClass = getattr(alg_module, config.alg.upper())
    alg = AlgClass(model, config, lambda: copy.deepcopy(model), tokenizer=tokenizer) # MEND    

    trainer = EditTrainer(alg, tokenizer, config, train_set, val_set)  # MEND, config, datasets -> Trainer
    trainer.run()


if __name__ == "__main__":
    run()
