import logging
import os
import shutil
import tempfile
import time
import json, pickle

import torch
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf

import wandb

from losses import kl_loc_loss
import utils
from utils import _logits, safe_backward, RunningStatAverager, EarlyStopper, formatted_timestamp, time_delta_seconds

from tqdm import tqdm

import copy


LOG = logging.getLogger(__name__)


class BaseTrainer:
    def __init__(self, model, config, train_set: Dataset, val_set: Dataset):
        self.model = model
        self.config = config

        if config.train_base:
            self.original_model = self.model.model_constructor()
            self.original_model.load_state_dict(self.model.model.state_dict())
            self.original_model.to(self.config.device)
        else:
            self.original_model = copy.deepcopy(self.model.model)

        self.model.to(self.config.device)

        self.train_set = train_set
        self.val_set = val_set

        if self.config.eval_only:
            # Eval once and quit
            self.config.max_iters = 0

        if not self.config.eval_only:
            self.OptimizerClass = getattr(torch.optim, config.opt)
            LOG.info(f"Building optimizer {self.OptimizerClass} with lr {config.lr}")
            self.opt = self.OptimizerClass(self.model.outer_parameters(), lr=config.lr)     # mend, edit_lrs

        if config.archive is not None:
            archive, config.archive = utils.load_archive(str(config.archive))
            self.model.load_state_dict(archive["model"])
            del archive["model"]
            if not self.config.eval_only:
                self.opt.load_state_dict(archive["opt"])
            del archive["opt"]

            self.archive = archive  # Save for later to load e.g. lr_opt params if they exist
        else:
            self.archive = None

        # outfiles
        tmpgetcwd = os.getcwd()
        with open(tmpgetcwd + "/config.json", "w") as f:
            json.dump(OmegaConf.to_container(config), f)
            print(f"Output the validation info to {tmpgetcwd}/config.json")
        model_dir = os.path.join(os.getcwd(), 'models')
        if not (self.config.debug and not self.config.save):
            os.makedirs(model_dir)
        run_date = os.getcwd().split('/')[-1]
        self.run_date = run_date
        safe_model_name = self.config.model.name.split("/")[-1]  # Make sure no slashes
        self.save_path = f"{model_dir}/{safe_model_name}.{run_date}"

        # if not (self.config.debug or self.config.eval_only):
        #     wandb_dir = tempfile.mkdtemp()
        #     wandb_name = f"{self.config.dataset} - {self.config.alg} - {safe_model_name} - {run_date}"
        #     if self.config.ref is not None:
        #         wandb_name += f" - {self.config.ref}"
        #     LOG.info(f"Writing wandb run \"{wandb_name}\" to {wandb_dir}")
        #     wandb.init(
        #         project="efk",
        #         # entity="patchable-lm",
        #         config=utils.flatten_dict(self.config),
        #         name=wandb_name,
        #         dir=wandb_dir,
        #         tags=[self.config.ref] if self.config.ref is not None else None
        #     )

        self.start_time = formatted_timestamp()

    def save_state(self, stats, global_iter):
        if (self.config.debug and not self.config.save) or self.config.eval_only:
            return

        obj = {
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
            "lr_opt": self.lr_opt.state_dict() if self.lr_opt is not None else None,
            "val_stats": stats,
            "start_time": self.start_time,
            "elapsed_time": time_delta_seconds(self.start_time),
            "step": self.global_iter
        }
        LOG.info(f"Saving model to {self.save_path}")

        if os.path.exists(self.save_path+"_"+str(global_iter)):
            bk_path = self.save_path+"_"+str(global_iter)+".bk"
            LOG.info(f"Moving old archive to {bk_path}")
            os.rename(self.save_path+"_"+str(global_iter), bk_path)

        torch.save(obj, self.save_path+"_"+str(global_iter))
        LOG.info("Write complete.")
        

    def echo(self, train_step, info_dict, pretty=False):
        if not self.config.silent:
            sep = "\n" if pretty else "; "

            def key_format(k):
                return k.ljust(20) if pretty else k
            LOG.info(f"Step {train_step}:")
            LOG.info(sep.join([f"{key_format(k)}: {v: 0.5f}" for k, v in info_dict.items()]))

    # def wandb_log(self, step, info_dict):
    #     if not (self.config.debug or self.config.eval_only):
    #         wandb.log(info_dict, step=step)

    def run(self):
        averager = RunningStatAverager("train")
        stopper = EarlyStopper(self.config.early_stop_patience, self.config.early_stop_key)
        self.global_iter = 0
        for global_iter in tqdm(range(0, self.config.max_iters)):
            self.global_iter = global_iter

            if not self.config.eval_only:
                train_info = self.train_step()
                averager.add(train_info)

                if global_iter % self.config.log_interval == 0:
                    avg_info = averager.average()
                    averager.reset()
                    self.echo(global_iter, avg_info)
                    # self.wandb_log(global_iter, avg_info)
            
            if global_iter % 5==0 and global_iter!=0:
                self.echo(global_iter, avg_info)
                if not self.config.eval_only:
                    self.save_state(avg_info, global_iter)
            if global_iter % self.config.val_interval == 0 and global_iter!=0:
                val_info = self.validate(steps=self.config.val_steps)
                self.echo(global_iter, val_info)
                # self.wandb_log(global_iter, val_info)
                if stopper.update(self.global_iter, val_info):
                    self.save_state(val_info, global_iter)  # New best
                    save_path = os.path.join(self.config.save_path, str(global_iter))
                    os.makedirs(save_path, exist_ok=True)
                    if not self.config.eval_only:
                        print(f"Save with the Huggingface form in {save_path}")
                        torch.save(self.config, os.path.join(save_path, "eval_args.bin"))
                        self.edited_model.save_pretrained(save_path)
                        self.tokenizer.save_pretrained(save_path)
                        torch.save(self.config, os.path.join(save_path, "training_args.bin"))
                        torch.save(self.opt.state_dict(), os.path.join(save_path, "optimizer.pt"))
                        torch.save(self.lr_opt.state_dict(), os.path.join(save_path, "lr_optimizer.pt"))
                    else:
                        print(f"Save with the Huggingface form in {save_path}")
                        torch.save(self.config, os.path.join(save_path, "eval_args.bin"))
                        self.edited_model.save_pretrained(save_path)
                        self.tokenizer.save_pretrained(save_path)
                        # torch.save(self.opt.state_dict(), os.path.join(save_path, "optimizer.pt"))
                        # torch.save(self.lr_opt.state_dict(), os.path.join(save_path, "lr_optimizer.pt"))
                    LOG.info("Write complete.")
                
                if stopper.should_stop():
                    LOG.info(f"No decrease in {self.config.early_stop_key} for {self.config.early_stop_patience} steps")
                    self.final_global_iter = global_iter
                    break

        if not self.config.eval_only:
            LOG.info(f"Training complete after {self.global_iter+1} steps.")

        if not self.config.eval.final_eval:
            return

        if not self.config.eval_only:
            if (not self.config.debug) or self.config.save:
                tmp_save_path = self.save_path+"_"+str(self.final_global_iter)
                archive = torch.load(tmp_save_path, map_location="cuda:0")
                LOG.info(f"Loading best model from step {archive['step']}, elapsed time {archive['elapsed_time']}")
                self.model.to("cuda:0")
                self.model.load_state_dict(archive["model"])
                self.model.to(self.config.device)

        val_steps = 200 if self.config.debug else None
        val_info = self.validate(log=True, steps=val_steps)
        self.echo(self.global_iter, val_info, pretty=True)
        # self.wandb_log(self.global_iter + self.config.val_interval, val_info)

        if self.config.results_dir is not None:
            results_path = f"{self.config.results_dir}/results_{self.run_date}.json"
            latest_path = f"{self.config.results_dir}/results_latest.json"
        else:
            results_path = f"{os.getcwd()}/results.json"
            latest_path = f"{os.getcwd()}/results_latest.json"

        with open(results_path, "w") as f:
            json.dump({"results": val_info, "config": OmegaConf.to_container(self.config)}, f)
            LOG.info("Wrote results to:")
            LOG.info(results_path)

        shutil.copy(results_path, latest_path)
        LOG.info("Copied to:")
        LOG.info(latest_path)



class EditTrainer(BaseTrainer):
    def __init__(self, model, tokenizer, config, train_set: Dataset, val_set: Dataset):
        super().__init__(model, config, train_set, val_set)

        self.edit_gen = self.train_set.edit_generator(batch_size=config.batch_size)
        if hasattr(model, "edit_lrs") and not self.config.eval_only:
            self.lr_opt = self.OptimizerClass([model.edit_lrs], config.lr_lr)
            if self.archive is not None:
                self.lr_opt.load_state_dict(self.archive["lr_opt"])
        else:
            self.lr_opt = None

        if hasattr(self.config, "ft"):
            if getattr(self.config.ft, "use_locality", False):
                batch = next(self.edit_gen)
                self.model.loc_ids = batch["loc"]["input_ids"]
                self.model.loc_masks = batch["loc"]["attention_mask"]
        
        self.tokenizer = tokenizer

    def edit_step(self, batch, training: bool):
        self.model.train(training)
        self.original_model.train(training)

        with torch.no_grad():
            base_output = self.model.model(**batch["loc"], return_dict=True)
            base_logits = base_output.logits

        # Do the edit
        start = time.time()
        edited_model, model_info = self.model.edit(batch["edit_inner"], batch["cond"])
        self.edited_model = edited_model.model
        edit_time = time.time() - start

        with torch.set_grad_enabled(training):
            # Editing loss
            post_edit_logits_anti, post_edit_logits_stereo = edited_model(**batch["edit_outer"]['anti']), edited_model(**batch["edit_outer"]['stereo'])
            l_edit = self.model.edit_loss_fn(
                post_edit_logits_anti, batch["edit_outer"]["anti"]["labels"], 
                post_edit_logits_stereo, batch["edit_outer"]["stereo"]["labels"]
            )["loss"]

            # Locality loss
            post_base_output = edited_model.model(**batch["loc"], return_dict=True)
            post_base_logits = post_base_output.logits
            kl_mask = batch["loc"].get("decoder_attention_mask", batch["loc"]["attention_mask"])
            l_loc = kl_loc_loss(base_logits.detach(), post_base_logits, mask=kl_mask)

        l_total_edit = self.config.cedit * l_edit + self.config.cloc * l_loc

        if training:
            safe_backward(l_total_edit, self.model.outer_parameters(), self.config.accumulate_bs)

        # Collect some useful metrics
        with torch.no_grad():
            post_edit_dict = self.model.edit_loss_fn(
                post_edit_logits_anti, batch["edit_outer"]["anti"]["labels"],
                post_edit_logits_stereo, batch["edit_outer"]["stereo"]["labels"])
            post_loc_dict = self.model.loc_loss_fn(post_base_output, batch["loc"]["labels"])
            pre_loc_dict = self.model.loc_loss_fn(base_output, batch["loc"]["labels"])
            # LMS
            lms=0
            total = 0
            for antis, locs in zip(post_edit_dict['anti_score'], post_loc_dict['loc_score']):
                total += 1
                if antis > locs:
                    lms += 1
            for stereos, locs in zip(post_edit_dict['stereo_score'], post_loc_dict['loc_score']):
                total += 1
                if stereos > locs:
                    lms += 1
            lms = lms / total

        info_dict = {}
        info_dict['lms'] = lms
        info_dict['loss/edit'] = l_edit.item()
        info_dict['loss/loc'] = l_loc.item()
        info_dict['edit/ss_score'] = post_edit_dict["ss_score"]
        info_dict['edit/anti_log_prob'] = post_edit_dict["anti_log_prob"].item()
        info_dict['edit/stereo_log_prob'] = post_edit_dict["stereo_log_prob"].item()
        info_dict["pre_loc/avg_log_prob"] = pre_loc_dict["avg_log_prob"]
        info_dict["nll/pre_loc"] = pre_loc_dict["loss"].item()
        info_dict["post_loc/avg_log_prob"] = post_loc_dict["avg_log_prob"].item()
        info_dict["nll/post_loc"] = post_loc_dict["loss"].item()
        info_dict["n_tokens/pre_loc"] = pre_loc_dict["n_tokens"]
        info_dict["n_tokens/post_loc"] = post_loc_dict["n_tokens"]
        info_dict["time/edit"] = edit_time

        # Base loss
        if self.config.train_base:
            with torch.no_grad():
                original_output = self.original_model(**batch["loc"], return_dict=True)
                original_logits = original_output.logits

                original_loc_dict = self.model.loc_loss_fn(original_logits, batch["loc"]["labels"])

            base_logits = self.model(**batch["loc"])
            l_base = kl_loc_loss(original_logits.detach(), base_logits, mask=kl_mask.detach())

            if training:
                safe_backward(l_base, self.model.outer_parameters(), self.config.accumulate_bs, allow_unused=True)

            info_dict['loss/base'] = l_base.item()
            info_dict['nll/original'] = original_loc_dict["loss"].item()
            info_dict['avg_log_prob/original'] = original_loc_dict["avg_log_prob"].item()
            info_dict["n_tokens/original"] = original_loc_dict["n_tokens"]
        else:
            l_base = torch.tensor(0.)

        l_total = l_total_edit + self.config.cbase * l_base

        info_dict["loss/total"] = l_total.item()
        info_dict["loss/total_edit"] = l_total_edit.item()
        info_dict["memory/alloc_max"] = torch.cuda.max_memory_allocated()
        info_dict["memory/res_max"] = torch.cuda.max_memory_reserved()
        info_dict = {**info_dict, **model_info}



        return l_total, l_edit, l_loc, l_base, info_dict

    def train_step(self):
        
        l_total, l_edit, l_loc, l_base, info_dict = self.edit_step(next(self.edit_gen), training=True)

        if self.global_iter > 0 and self.global_iter % self.config.accumulate_bs == 0:
            grad = torch.nn.utils.clip_grad_norm_(self.model.outer_parameters(), self.config.grad_clip,
                                                  error_if_nonfinite=True)
            info_dict['grad'] = grad.item()

            self.opt.step()
            self.opt.zero_grad()

            if self.lr_opt is not None:
                self.lr_opt.step()
                self.lr_opt.zero_grad()

                for lr_idx, lr in enumerate(self.model.edit_lrs):
                    info_dict[f'lr/lr{lr_idx}'] = lr.item()

        return info_dict

    def _inline_validation_log(self, step, stats, start_time, steps):
        elapsed = (time.time() - start_time) / (step + 1)
        prog = f"{step+1}/{steps}".ljust(20)
        acc = f"{stats['edit/ss_score_val']:<12.5f}"
        
        if self.config.task in ["fc", "qa"] or "debias" in self.config.task:
            draw_pre = f"{stats['pre_loc/avg_log_prob_val']:<12.5f}"
            draw_post = f"{stats['post_loc/avg_log_prob_val']:<12.5f}"
            draw_diff = f"{stats['pre_loc/avg_log_prob_val']-stats['post_loc/avg_log_prob_val']:<12.5f}"
            dn = "log_prob"  # drawdown name
        elif self.config.task in ["gen"]:
            draw_pre = f"{stats['perplexity/pre_loc_val']:<12.5f}"
            draw_post = f"{stats['perplexity/post_loc_val']:<12.5f}"
            draw_diff = f"{stats['perplexity/post_loc_val']-stats['perplexity/pre_loc_val']:<12.5f}"
            dn = "ppl"  # drawdown name
        else:
            raise RuntimeError(f"Didn't recognize task {self.config.task}")
        
        save_path = os.path.join(self.config.save_path, f"val_log_{step}")
        os.makedirs(save_path, exist_ok=True)
        
        if not self.config.eval_only:
            torch.save(self.config, os.path.join(save_path, "training_args.bin"))
            torch.save(self.opt.state_dict(), os.path.join(save_path, "optimizer.pt"))
            torch.save(self.lr_opt.state_dict(), os.path.join(save_path, "lr_optimizer.pt"))
        else:
            print(f"Save with the Huggingface form in {save_path}")
            self.edited_model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            torch.save(self.config, os.path.join(save_path, "eval_args.bin"))
        LOG.info("Write complete.")

        LOG.info(f"Step {prog} edit: {acc} {dn}_pre: {draw_pre} {dn}_post: {draw_post} {dn}_delta: {draw_diff} it_time: {elapsed:.4f}")

    def validate(self, steps=None, log: bool = False):
        if steps is None or steps > len(self.val_set):
            steps = len(self.val_set)

        if log:
            LOG.info(f"Beginning evaluation for {steps} steps...")
        averager = RunningStatAverager("val")
        val_edit_gen = self.val_set.edit_generator(batch_size=self.config.val_batch_size, n=steps)
        os.makedirs(self.save_path, exist_ok=True)
        start_time = time.time()
        for val_step in tqdm(range(steps), desc="Validation"):
            _, _, _, _, info_dict = self.edit_step(next(val_edit_gen), training=False)
            averager.add(info_dict)
            pickle.dump(info_dict, open(f"{self.save_path}/val_log_{val_step}.pk", "wb"))
            print(f"One batch validation completes. Save in {self.save_path}/val_log_{val_step}.pk")
            if log and self.config.eval.verbose and (val_step + 1) % self.config.eval.log_interval == 0:
                self._inline_validation_log(val_step, averager.average(), start_time, steps)

        if log and self.config.eval.verbose:
            self._inline_validation_log(val_step, averager.average(), start_time, steps)
        elapsed = time.time() - start_time
        stats = averager.average()
        stats["eval_time/elapsed"] = elapsed
        stats["eval_time/average"] = elapsed / steps

        return stats
