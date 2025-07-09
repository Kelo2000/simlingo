import os
from pathlib import Path
from typing import List

import hydra
import torch
from omegaconf import OmegaConf
from transformers import AutoProcessor

from simlingo_training.config import TrainConfig
from simlingo_training.callbacks.visualise import visualise_waypoints
from simlingo_training.utils.custom_types import (
    DrivingExample,
    DrivingInput,
    DrivingLabel,
    LanguageLabel,
)


def compute_metrics(pred: torch.Tensor, gt: torch.Tensor, step_sec: float = 0.2) -> dict:
    """Compute ADE/FDE and L2 errors at 1s/2s/3s if available."""
    min_len = min(pred.size(1), gt.size(1))
    pred = pred[:, :min_len]
    gt = gt[:, :min_len]
    err = torch.norm(pred - gt, dim=-1)
    ade = err.mean().item()
    fde = err[:, -1].mean().item()
    metrics = {"ADE": ade, "FDE": fde}
    for t in [1.0, 2.0, 3.0]:
        idx = int(round(t / step_sec))
        if idx < err.size(1):
            metrics[f"L2_{int(t)}s"] = err[:, idx].mean().item()
    return metrics


def _move_language_label(label: LanguageLabel, device: torch.device) -> LanguageLabel:
    """Move tensor fields of a LanguageLabel to the given device."""
    def maybe(t):
        return t.to(device) if t is not None else None

    return LanguageLabel(
        phrase_ids=maybe(label.phrase_ids),
        phrase_valid=maybe(label.phrase_valid),
        phrase_mask=maybe(label.phrase_mask),
        placeholder_values=label.placeholder_values,
        language_string=label.language_string,
        loss_masking=maybe(label.loss_masking),
    )


def move_example_to_device(example: DrivingExample, device: torch.device) -> DrivingExample:
    inp = example.driving_input
    lbl = example.driving_label
    driving_input = DrivingInput(
        camera_images=inp.camera_images.to(device),
        image_sizes=inp.image_sizes.to(device),
        camera_intrinsics=inp.camera_intrinsics.to(device),
        camera_extrinsics=inp.camera_extrinsics.to(device),
        vehicle_speed=inp.vehicle_speed.to(device),
        target_point=inp.target_point.to(device),
        prompt=_move_language_label(inp.prompt, device),
        prompt_inference=_move_language_label(inp.prompt_inference, device),
    )
    driving_label = DrivingLabel(
        waypoints=lbl.waypoints.to(device),
        path=lbl.path.to(device),
        answer=_move_language_label(lbl.answer, device),
        image_ff_org=lbl.image_ff_org.to(device),
        eval_infos=lbl.eval_infos,
    )
    return DrivingExample(
        driving_input=driving_input,
        driving_label=driving_label,
        run_id=example.run_id,
        qa_templates=example.qa_templates,
    )


@hydra.main(config_path="../config", config_name="config", version_base="1.1")
def main(cfg: TrainConfig) -> None:
    """Run evaluation on the CoVLA dataset."""
    torch.set_float32_matmul_precision("high")

    processor = AutoProcessor.from_pretrained(cfg.model.vision_model.variant, trust_remote_code=True)

    dm = hydra.utils.instantiate(
        cfg.data_module,
        processor=processor,
        encoder_variant=cfg.model.vision_model.variant,
        llm_variant=cfg.model.language_model.variant,
        predict=False,
        _recursive_=False,
    )
    dm.setup()
    dl = dm.val_dataloader()

    model = hydra.utils.instantiate(
        cfg.model,
        cfg_data_module=cfg.data_module,
        processor=processor,
        cache_dir=None,
        _recursive_=False,
    )
    if not cfg.checkpoint:
        raise ValueError("cfg.checkpoint must be set to a model checkpoint")
    state_dict = torch.load(cfg.checkpoint, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    pred_all: List[torch.Tensor] = []
    gt_all: List[torch.Tensor] = []

    viz = cfg.visualize
    viz_dir = Path("visualisations")
    if viz:
        viz_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for idx, batch in enumerate(dl):
            batch = move_example_to_device(batch, device)
            pred_wps, _, _ = model.forward(batch, return_language=True)
            pred_all.append(pred_wps.cpu())
            gt_all.append(batch.driving_label.waypoints.cpu())

            if viz and idx < 50:  # cap visualisations
                fig, txt = visualise_waypoints(batch, pred_wps.cpu())
                fig_path = viz_dir / f"batch_{idx:04d}.png"
                txt.save(viz_dir / f"batch_{idx:04d}_text.png")
                from PIL import Image

                Image.fromarray(fig).save(fig_path)

    pred_all = torch.cat(pred_all, dim=0)
    gt_all = torch.cat(gt_all, dim=0)
    metrics = compute_metrics(pred_all, gt_all)
    print(OmegaConf.to_yaml(metrics))


if __name__ == "__main__":
    main()

