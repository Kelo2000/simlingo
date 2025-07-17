import datetime
import json
import os
import random
from pathlib import Path
from pprint import PrettyPrinter
from typing import Dict, Optional, Tuple, List

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from hydra.utils import get_original_cwd


from simlingo_training.models.adaptors.adaptors import DrivingAdaptor, LanguageAdaptor, WaypointInputAdaptor, AdaptorList
from simlingo_training.models.utils import summarise_losses
from simlingo_training.utils.custom_types import (DrivingExample, DrivingInput,
                                                DrivingLabel, DrivingOutput,
                                                TrainingOutput)


pprint = PrettyPrinter().pprint

def decode_uint8(encoded: torch.Tensor) -> List[str]:
    return [row.tobytes().decode("utf-8").rstrip("\0") for row in encoded.cpu().numpy()]

class NormZeroOne(nn.Module):
    def __init__(self, min_max: Tuple[float, float]):
        super().__init__()
        self.register_buffer("min_max", torch.tensor(min_max, dtype=torch.float), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """Normalise tensor to [0, 1] using values from min_max"""
        return (x - self.min_max[0]) / (self.min_max[1] - self.min_max[0])


class DrivingModel(pl.LightningModule):
    def __init__(
        self,
        cfg_data_module,
        processor,
        cache_dir,
        **cfg,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        for key, value in cfg.items():
            setattr(self, key, value)
            
        self.processor = processor
        
        self.prediction = {}
        
        self.predict_language = True
        
        self.cfg_data_module = cfg_data_module
        
        self.vision_model = hydra.utils.instantiate(
            self.vision_model,
            cfg_data_module=cfg_data_module,
            processor=self.processor,
            cache_dir=cache_dir,
            _recursive_=False
        )
            
        self.language_model = hydra.utils.instantiate(
            self.language_model,
            cache_dir=cache_dir,
            _recursive_=False
        )

        self.all_predictions = {}
        self.all_losses = {}
        
        driving = None
        driving = DrivingAdaptor(
            self.language_model.hidden_size,
            speed_wps_mode=self.speed_wps_mode,
        )

        self.adaptors = AdaptorList(
            language=LanguageAdaptor(self.language_model),
            driving=driving,
        )

        self.wp_encoder = WaypointInputAdaptor(
            token_size=self.language_model.hidden_size,
            hidden_size=256,
            hidden_size2=512,
            # norm_layer=NormZeroOne(min_max=(-32.0, 32.0)),
        )

        if 'tokenizer' in self.processor.__dict__:
            self.tokenizer = self.processor.tokenizer
        else:
            self.tokenizer = self.processor


    def forward(self,
        example: DrivingExample,
        return_language: Optional[bool] = None,
        prompt_ids: Optional[Tensor] = None,
    ) -> DrivingOutput:
        """
        Samples a trajectory from the model.
        """
        self.speed_wps, self.language = None, []
        try:
            driving_input = example.driving_input
        except AttributeError:
            driving_input = example
        
        if driving_input is not None:
            adaptor_dict = self.adaptors(example, inference=True)
            adaptor_dict = self.vision_model.image_encoder.replace_placeholder_tokens(
                    adaptor_dict = adaptor_dict,
                    pixel_values = driving_input.camera_images,
                    placeholder_values = driving_input.prompt_inference.placeholder_values,
                    wp_encoder = self.wp_encoder,
                )
            
            input_embeds_all = adaptor_dict["language_inputs"]
            attention_masks = adaptor_dict['language_inputs_mask']


        if self.predict_language:
            # per batch item because of padding
            for b_idx, (input_embed, attention_mask) in enumerate(zip(input_embeds_all, attention_masks)):
                input_embed = input_embed.unsqueeze(0)
                attention_mask = attention_mask.unsqueeze(0)
                if self.language_model.variant == 'OpenGVLab/InternVL2-4B':
                    eos = self.tokenizer.added_tokens_encoder['<|end|>']
                elif self.language_model.variant == 'OpenGVLab/InternVL2-2B':
                    eos = self.tokenizer.added_tokens_encoder['<|im_end|>']
                else:
                    eos = self.tokenizer.eos_token_id

                # BUG: input_embeds, cot
                sampled_tokens, input_embeds = self.language_model.greedy_sample(
                    input_embed,
                    eos_token_id=eos,
                    max_new_tokens=100,
                    input_embed_matrix=self.adaptors.language.embed_tokens.weight,
                    logit_matrix=self.adaptors.language.lm_head.weight,
                    attention_mask=attention_mask,
                    # position_ids=position_ids,
                )
                
                inputs_driving = self.adaptors.driving(driving_input)
                input_embed_concat = torch.cat((input_embeds, inputs_driving["inputs"][b_idx].unsqueeze(0)), dim=1)
                features, logits = self.language_model.forward(input_embed_concat)

                len_driving = inputs_driving["inputs"].size(1)

                driving_features = features[:, -len_driving:]
                driving_logits = logits[:, -len_driving:]
                predictions = self.adaptors.driving.get_predictions(driving_features, driving_logits)
                    
                for k, v in predictions.items():
                    if v is not None:
                        if hasattr(self, k) and getattr(self, k) is not None:
                            if isinstance(v, torch.Tensor):
                                setattr(self, k, torch.cat((getattr(self, k), v), dim=0))
                            elif isinstance(v, list):
                                getattr(self, k).append(v)
                            else:
                                raise NotImplementedError(f"Type of {k} not supported")
                        else:
                            setattr(self, k, v)
                                
                self.language.append(self.tokenizer.batch_decode(sampled_tokens, skip_special_tokens=True)[0])
        else:
            # single forward pass same as during training so we can use the same function
            features = self.forward_model(driving_input, adaptor_dict)
            outputs_by_adaptor = self.adaptors.split_outputs_by_adaptor(adaptor_dict, features)
            predictions = self.adaptors.driving.get_predictions(outputs_by_adaptor['driving'])

            for k, v in predictions.items():
                if v is not None:
                    setattr(self, k, v)

        return self.speed_wps, self.language


    def forward_model(self, 
                      driving_input: DrivingInput, 
                      adaptor_dict: Dict, 
                      driving_labels: DrivingLabel = None,
                    #   language_embeds: Tensor = None
                      ) -> Tensor:
        """
        Forward model conditioned on the given driving input.
        """
        
        adaptor_dict = self.vision_model.image_encoder.replace_placeholder_tokens(
            adaptor_dict = adaptor_dict,
            pixel_values = driving_input.camera_images,
            placeholder_values = driving_input.prompt.placeholder_values,
            wp_encoder = self.wp_encoder,
        )

        position_ids = None
        adaptor_embeds = adaptor_dict["inputs"]
        adaptor_mask = adaptor_dict['inputs_mask']

        input_embeds = adaptor_embeds
        input_embeds = input_embeds.to(
            dtype=self.language_model.model.dtype
        )
        attention_mask = adaptor_mask

        outputs = self.language_model.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=input_embeds,
            output_hidden_states=True,
            return_dict=True,
        )
        features = outputs.hidden_states[-1]
        logits = outputs[0]

        vision_features, adaptor_features = features.split(
            [features.size(1) - adaptor_embeds.size(1), adaptor_embeds.size(1)], dim=1
        )
        vision_logits, adaptor_logits = logits.split(
            [logits.size(1) - adaptor_embeds.size(1), adaptor_embeds.size(1)], dim=1
        )
        return adaptor_features, adaptor_logits
    

    def forward_loss(self, example: DrivingExample, per_sample=False) -> TrainingOutput:
        """
        Forward pass of the model for a driving input, followed by
        computing the next token cross-entropy loss.

        Args:
            driving_input: input to the vision encoder.
            text_ids: Text ids tensor of shape [B, T]. These are input to the model and used in the loss.
            text_mask: Text mask tensor of shape [B, T].
        """

        adaptor_dict = self.adaptors(example)
        adaptor_embeds = adaptor_dict["inputs"]
        adaptor_mask = adaptor_dict['inputs_mask']

        adaptor_features, adaptor_logits = self.forward_model(example.driving_input, adaptor_dict, driving_labels=example.driving_label)
        loss_dict = self.adaptors.compute_loss(adaptor_features, adaptor_logits, adaptor_dict, example)

        loss_dict_only_losses = {k:v for k, v in loss_dict.items() if k.endswith("loss")}
        loss_logs = {k:v for k, v in loss_dict.items() if k.endswith("log")}
        
        pred_labels = {k:v for k, v in loss_dict.items() if not k.endswith("loss") and not k.endswith("log")}
        if per_sample:
            return loss_dict_only_losses, pred_labels

        return summarise_losses(loss_dict_only_losses), loss_logs

    def training_step(self, batch: DrivingExample, _batch_idx: int = 0):
        output, loss_logs = self.forward_loss(batch)
        logs = output #.update(loss_logs)
        self.log_training_output(logs, "train")

        # log the loss
        self.log("train/loss", output.loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {"loss": output.loss, "outputs": output}


    def validation_step(self, batch: DrivingExample, _batch_idx: int = 0):
        
        output, loss_logs = self.forward_loss(batch)
        logs = output #.update(loss_logs)
        self.log_training_output(logs, "val")

        # log the loss
        self.log("val/loss", output.loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {"loss": output.loss, "outputs": output}

    def predict_step(self, batch: DrivingExample, _batch_idx: int = 0):
        run_ids = decode_uint8(batch.run_id)
        
        speed_wps, language = self.forward(batch, return_language=True)
        speed_wps_gt = batch.driving_label.waypoints
        language_gt = batch.driving_label.answer.language_string
        
        if len(self.prediction) == 0:
            self.prediction = {
                "waypoints": [speed_wps],
                "language": language,
                "waypoints_gt": [speed_wps_gt],
                "language_gt": language_gt,
                "prompt": batch.driving_input.prompt.language_string,
                "path": run_ids,
                "qa_templates": batch.qa_templates if batch.qa_templates is not None else [],
                "eval_infos": batch.driving_label.eval_infos if batch.driving_label.eval_infos is not None else [],
            }
        else:
            self.prediction["waypoints"].append(speed_wps)
            self.prediction["language"].extend(language)
            self.prediction["waypoints_gt"].append(speed_wps_gt)
            self.prediction["language_gt"].extend(language_gt)
            self.prediction["prompt"].extend(batch.driving_input.prompt.language_string)
            self.prediction["path"].extend(run_ids)
            if batch.qa_templates is not None:
                self.prediction["qa_templates"].extend(batch.qa_templates)
            if batch.driving_label.eval_infos is not None:
                self.prediction["eval_infos"].extend(batch.driving_label.eval_infos)
            
        
        return speed_wps, language, speed_wps_gt, language_gt

    def on_predict_epoch_end(self) -> None:    

        repo_path = get_original_cwd()

        if self.trainer.ckpt_path is not None:
            ckpt_path = Path(self.trainer.ckpt_path).parent.parent
        else:
            ckpt_path = Path(f'{repo_path}/outputs/{self.language_model.variant}')
        save_prediction_path = ckpt_path / "predictions"
        save_prediction_path.mkdir(exist_ok=True, parents=True)
        
        samples_cot = [i for i, l in enumerate(self.prediction["prompt"]) if "What should the ego do next?" in l]
        samples_qa = [i for i, l in enumerate(self.prediction["prompt"]) if "Q:" in l]
        samples_all = [i for i in range(len(self.prediction["prompt"]))]
        language = [(l, l_gt, p) for l, l_gt, p in zip(self.prediction["language"], self.prediction["language_gt"], self.prediction["path"])]
        
        if len(samples_qa) > 0 and self.prediction.get("qa_templates"):
            # sort by templates
            sorted_samples = {} # question: {answer: [language, language_gt]}
            for qa_template, language_sample in zip(self.prediction["qa_templates"], language):
                if qa_template is None:
                    continue
                question = qa_template[0]
                answer = qa_template[1]
                if question not in sorted_samples:
                    sorted_samples[question] = {}
                if answer not in sorted_samples[question]:
                    sorted_samples[question][answer] = []
                sorted_samples[question][answer].append(language_sample)
            
            if os.path.exists(f"{str(save_prediction_path)}/sorted_qa_templates_rank_{self.local_rank}.json"):
                time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                with open(f"{str(save_prediction_path)}/sorted_qa_templates_rank_{self.local_rank}_{time}.json", "w") as f:
                    json.dump(sorted_samples, f, indent=4)
            else:
                with open(f"{str(save_prediction_path)}/sorted_qa_templates_rank_{self.local_rank}.json", "w") as f:
                    json.dump(sorted_samples, f, indent=4)
        
        for samples, name in zip([samples_cot, samples_qa, samples_all], ["cot", "qa", "all"]):
            language_samples = [l for i, l in enumerate(language) if i in samples]
        
            # save language predictions
            save_path_tmp = f"{str(save_prediction_path)}/language_preds_{name}_rank_{self.local_rank}.json"
            if os.path.exists(save_path_tmp):
                time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                save_path_tmp = f"{str(save_prediction_path)}/language_preds_{name}_rank_{self.local_rank}_{time}.json"
            with open(save_path_tmp, "w") as f:
                json.dump(language_samples, f, indent=4)
            
        waypoints_preds = self.prediction["waypoints"]
        waypoints_preds = torch.cat(waypoints_preds, dim=0)
        waypoints_gt = self.prediction["waypoints_gt"]
        waypoints_gt = torch.cat(waypoints_gt, dim=0)
        err = torch.norm(waypoints_preds - waypoints_gt, dim=-1)
        ade = err.mean().item()
        fde = err[:, -1].mean().item()
        metrics = {"ADE": ade, "FDE": fde}
        save_path_tmp = f"{str(save_prediction_path)}/results_rank_{self.local_rank}.json"
        if os.path.exists(save_path_tmp):
            time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_path_tmp = f"{str(save_prediction_path)}/results_rank_{self.local_rank}_{time}.json"
        with open(save_path_tmp, "w") as f:
            json.dump(metrics, f, indent=4)
        

    def log_training_output(self, training_output: TrainingOutput, mode: str, dataset: Optional[str] = None):
        losses = {k: n.detach() for k, n in training_output.loss_averages.items()}
        counts = {k: n.detach().sum() for k, n in training_output.loss_counts.items()}
        losses["loss"] = training_output.loss.detach()
        counts["loss"] = 1  # loss is already averaged
        for k, v in sorted(losses.items()):
            log_key = f"{mode}_losses/{k}"
            self.log(log_key, v, batch_size=counts[k], sync_dist=True, add_dataloader_idx=False)


    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=self.betas,
        )
        if self.trainer.max_steps == -1:
            max_steps = self.trainer.estimated_stepping_batches
        else:
            max_steps = self.trainer.max_steps
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.lr, total_steps=max_steps, pct_start=self.pct_start, verbose=False
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "frequency": 1, "interval": "step"}}
