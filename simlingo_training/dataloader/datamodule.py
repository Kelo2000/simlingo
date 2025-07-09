# Standard library imports
import itertools
from typing import List

# Third-party imports
import hydra
import line_profiler
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoProcessor

# Local/project specific imports
# from simlingo_training.dataloader.dataset_driving import Data_Driving # is called directly by hydra.utils.instantiate, keeping here to make it easier to find
# from simlingo_training.dataloader.dataset_dreamer import Data_Dreamer # is called directly by hydra.utils.instantiate, keeping here to make it easier to find
from simlingo_training.utils.custom_types import DrivingExample, DrivingInput, DrivingLabel, LanguageLabel
from simlingo_training.utils.internvl2_utils import preprocess_image_batch, get_custom_chat_template, get_num_image_tokens_per_patch
from simlingo_training.utils.projection import get_camera_intrinsics, get_camera_extrinsics

def encode_uint8(strings: List[str], common_length: int) -> torch.Tensor:
    max_len = max(len(s) for s in strings)
    assert max_len <= common_length, f"String is too long: {max_len} > {common_length}"
    padded_strings = [s.ljust(common_length, '\0') for s in strings]
    return torch.tensor([bytearray(s, 'utf-8') for s in padded_strings], dtype=torch.uint8)


class DataModule(LightningDataModule):
    def __init__(
        self,
        base_dataset,
        processor,
        predict=False,
        **cfg,
    ):
        super().__init__()
        for key, value in cfg.items():
            setattr(self, key, value)
            
        for key, value in base_dataset.items():
            setattr(self, key, value)
            
        self.cfg = cfg
        self.base_dataset = base_dataset
        self.processor = processor
        self.predict = predict
        
        self.printed = False

        self.NUM_IMAGE_PATCHES = 2
        self.IMAGES_TO_CONSIDER = ['image_ff'] # front-forward image, other images are not supported
        # taken from:
        # https://github.com/OpenGVLab/InternVL/blob/9d3a709b16874e73ffdd38b9cf53296fae4589b9/internvl_chat/internvl/train/constants.py#L7
        # https://github.com/OpenGVLab/InternVL/blob/9d3a709b16874e73ffdd38b9cf53296fae4589b9/internvl_chat/internvl/model/internvl_chat/modeling_internvl_chat.py#L294
        self.IMG_START_TOKEN='<img>'
        self.IMG_END_TOKEN='</img>'
        self.IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'

        self.num_image_tokens_per_patch = get_num_image_tokens_per_patch(self.encoder_variant)
        self.num_image_tokens_total = self.num_image_tokens_per_patch * self.NUM_IMAGE_PATCHES
            
        # add <WAYPOINT> token
        if 'tokenizer' in self.processor.__dict__:
            self.tokenizer = self.processor.tokenizer
        else:
            self.tokenizer = self.processor
        # TODO: not needed anymore?
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['<WAYPOINTS>','<WAYPOINTS_DIFF>', '<ORG_WAYPOINTS_DIFF>', '<ORG_WAYPOINTS>', '<WAYPOINT_LAST>', '<ROUTE>', '<ROUTE_DIFF>', '<TARGET_POINT>']})
        self.tokenizer.padding_side = "left"

    def setup(self, stage=None):
        if not self.predict:
            if getattr(self, "covla_dataset", None) is None:
                raise ValueError("CoVLA dataset configuration missing")

            self.train_dataset = hydra.utils.instantiate(
                self.covla_dataset,
                split="train",
                **self.cfg,
                **self.base_dataset,
                _recursive_=False,
            )
            self.val_dataset = hydra.utils.instantiate(
                self.covla_dataset,
                split="val",
                **self.cfg,
                **self.base_dataset,
                _recursive_=False,
            )

            self.sampler_train = None
            self.predict_dataset = None
            self.val_datasets = [self.val_dataset]
        else:
            if self.qa_dataset is not None:
                predict_dataset = self.qa_dataset
            elif self.insteval_dataset is not None:
                predict_dataset = self.insteval_dataset
            else:
                predict_dataset = self.covla_dataset

            self.predict_dataset = hydra.utils.instantiate(
                predict_dataset,
                split="val",
                **self.cfg,
                **self.base_dataset,
                _recursive_=False,
            )

        if not self.predict:
            self.val_dataset = self.val_datasets[0]


    def train_dataloader(self):
        if self.train_dataset is None:
            return None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.sampler_train is None,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=self.dl_collate_fn,
            sampler=self.sampler_train,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=self.dl_collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=self.dl_collate_fn,
            pin_memory=True,
        )


    @line_profiler.profile
    def dl_collate_fn(self, data):
        BS = len(data)
        grid_nums = [self.NUM_IMAGE_PATCHES] # we split the front forward into two patches (1x2)

        image_ff_pixel, image_ff_sizes = None, None
        image_ff_org = torch.tensor(np.asarray([data[i].image_ff_org_size for i in range(BS)]))
            
        for idx, img_to_consider in enumerate(self.IMAGES_TO_CONSIDER):
            img_tmp = getattr(data[0], img_to_consider)
            T, C, H, W = img_tmp.shape
            assert T == 1, "Only one timestep as input supported"
            
            images_batch_tensor = torch.tensor(np.asarray([getattr(data[i], img_to_consider) if getattr(data[i], img_to_consider) is not None else np.zeros_like(img_tmp) for i in range(len(data))])).float()
            images_batch_tensor = images_batch_tensor.view(BS*T, C, H, W)
            images_batch_list = list(images_batch_tensor)

            if 'internvl2' in self.encoder_variant.lower():
                # get image patches
                images_processed = preprocess_image_batch(images_batch_list, input_size=448, use_global_img=self.use_global_img, max_num_grid=grid_nums[idx])    
            else:
                raise ValueError(f"Image preprocessing for {self.encoder_variant} not implemented")
                
            images_pixel = images_processed['pixel_values']
            image_sizes = images_processed['image_sizes']
            
            assert images_pixel.shape[0] == BS * T
            num_patches = images_pixel.shape[1]
            assert images_pixel.shape[2] == C
            new_height = images_pixel.shape[3]
            new_width = images_pixel.shape[4]
            images_pixel = images_pixel.view(BS, T, num_patches, C, new_height, new_width)
            
            if img_to_consider == 'image_ff':
                image_ff_pixel = images_pixel
                image_ff_sizes = image_sizes
            else:
                raise ValueError(f"Image type {img_to_consider} not supported")

        conversations = [data[i].conversation for i in range(BS)]
        conversation_dict, question_dict = get_custom_chat_template(conversations, self.tokenizer, self.encoder_variant, self.num_image_tokens_total)

        placeholder_batch_list = []
        for i in range(BS):
            tmp = {}
            for key, value in data[i].placeholder_values.items():
                token_nr_key = self.tokenizer.convert_tokens_to_ids(key)
                tmp[token_nr_key] = value
            placeholder_batch_list.append(tmp)
                
        prompt_languagelabel = LanguageLabel(
            phrase_ids=conversation_dict['phrase_ids'],
            phrase_valid=conversation_dict['phrase_valid'],
            phrase_mask=conversation_dict['phrase_mask'],
            placeholder_values=placeholder_batch_list,
            language_string=conversation_dict['language_string'],
            loss_masking=conversation_dict['loss_masking'],
        )

        prompt_question_languagelabel = LanguageLabel(
            phrase_ids=question_dict['phrase_ids'],
            phrase_valid=question_dict['phrase_valid'],
            phrase_mask=question_dict['phrase_mask'],
            placeholder_values=placeholder_batch_list,
            language_string=question_dict['language_string'],
            loss_masking=question_dict['loss_masking'],
        )
        answer_string_list = [data[i].answer[0]['content'][0]['text'] for i in range(BS)]
        answer_label =  LanguageLabel(
            phrase_ids=None,
            phrase_valid=None,
            phrase_mask=None,
            placeholder_values=None,
            language_string=answer_string_list,
            loss_masking=None,
        )
        
        if self.base_dataset.use_1d_wps:
            waypoints = torch.tensor(np.asarray([data[i].waypoints_1d for i in range(len(data))])).float() # [B, F, 2] 11 future waypoints 0.2s apart
        else:
            waypoints = torch.tensor(np.asarray([data[i].waypoints for i in range(len(data))])).float() # [B, F, 2] 11 future waypoints 0.2s apart
        
        if self.predict:
            qa_templates = [data[i].qa_templates[0] if data[i].qa_templates is not None else None for i in range(BS) ]
            eval_infos = [data[i].eval_infos if data[i].eval_infos is not None else None for i in range(BS) ]
        else:
            qa_templates = None
            eval_infos = None
        
        target_points = []
        for d in data:
            if d.target_points is None:
                target_points.append(np.zeros(2))
            else:
                target_points.append(d.target_points)

        paths = []
        for d in data:
            if d.path is None:
                paths.append(np.zeros((self.base_dataset.num_route_points, 2)))
            else:
                paths.append(d.path)

        cam_ints = []
        cam_exts = []
        default_K = get_camera_intrinsics(W, H, 110).numpy()
        default_E = get_camera_extrinsics().numpy()
        for d in data:
            if getattr(d, "camera_intrinsics", None) is not None:
                cam_ints.append(d.camera_intrinsics)
            else:
                cam_ints.append(default_K)
            if getattr(d, "camera_extrinsics", None) is not None:
                cam_exts.append(d.camera_extrinsics)
            else:
                cam_exts.append(default_E)

        cam_ints = torch.tensor(np.asarray(cam_ints)).float()
        cam_exts = torch.tensor(np.asarray(cam_exts)).float()

        driving_input = DrivingInput(
            camera_images=image_ff_pixel,  # [B, T, N, C, H, W] uint8 [0, 255]
            image_sizes=image_ff_sizes,
            camera_intrinsics=cam_ints,
            camera_extrinsics=cam_exts,
            vehicle_speed=torch.tensor(
                np.asarray([d.speed for d in data])
            ).float(),  # [B, S] float32
            target_point=torch.tensor(np.asarray(target_points)).float(),
            prompt=prompt_languagelabel,
            prompt_inference=prompt_question_languagelabel,
        )

        driving_label = DrivingLabel(
            waypoints=waypoints,
            path=torch.tensor(np.asarray(paths)).float(),
            answer=answer_label,
            image_ff_org=image_ff_org,
            eval_infos=eval_infos,
        )
            
        return DrivingExample(
            driving_input=driving_input,
            driving_label=driving_label,
            run_id=encode_uint8([data[i].measurement_path for i in range(BS)], 1000),  # [B] str
            qa_templates=qa_templates,
        )

    def dl_collate_fn_val(self, data):
        pass

    def dl_collate_fn_test(self, data):
        pass


@hydra.main(config_path=f"../config", config_name="config", version_base="1.1")
def test(cfg):
    
    get_waypoint_stats = True
    
        
    processor = AutoProcessor.from_pretrained(cfg.model.vision_model.variant, trust_remote_code=True)
    dm = hydra.utils.instantiate(
        cfg.data_module,
        processor=processor,
        # tokenizer=llm_tokenizer,
        encoder_variant=cfg.model.vision_model.variant,
        llm_variant="llava-hf/llava-v1.6-mistral-7b-hf",
        _recursive_=False
    )


    dm.setup()
    dl = dm.val_dataloader()
    print(dl.dataset.__len__())

    iterations = 0
    
    all_waypoints = []
    all_waypoints_diff = []
    for batch in dl:
        
        iterations += 1
        
        if iterations % 100 == 0:
            print(f"Iteration: {iterations}")
        
        if iterations > 20000:
            break
        
        if get_waypoint_stats:
            # get stats about range of waypoints
            waypoints = batch.driving_label.waypoints
            all_waypoints.append(waypoints)
            
            # get residuals
            residuals = waypoints[:,1:] - waypoints[:,:-1]
            all_waypoints_diff.append(residuals)
            
    # get histogram of waypoints
    if get_waypoint_stats:
        all_waypoints = torch.cat(all_waypoints, dim=0)
        all_waypoints_diff = torch.cat(all_waypoints_diff, dim=0)
        
        all_waypoints = all_waypoints.view(-1, 2)
        all_waypoints_diff = all_waypoints_diff.view(-1, 2)
        
        
        
        import matplotlib.pyplot as plt
        plt.hist(all_waypoints[:,0].numpy(), bins=100)
        plt.savefig('waypoints_x.png')
        max_x = all_waypoints[:,0].max().item()
        min_x = all_waypoints[:,0].min().item()
        print(f"Max x: {max_x}, Min x: {min_x}")
        plt.clf()
        plt.hist(all_waypoints[:,1].numpy(), bins=100)
        plt.savefig('waypoints_y.png')
        max_y = all_waypoints[:,1].max().item()
        min_y = all_waypoints[:,1].min().item()
        print(f"Max y: {max_y}, Min y: {min_y}")
        plt.clf()
        
        plt.hist(all_waypoints_diff[:,0].numpy(), bins=100)
        plt.savefig('waypoints_diff_x.png')
        max_x_diff = all_waypoints_diff[:,0].max().item()
        min_x_diff = all_waypoints_diff[:,0].min().item()
        print(f"Max x diff: {max_x_diff}, Min x diff: {min_x_diff}")
        plt.clf()
        plt.hist(all_waypoints_diff[:,1].numpy(), bins=100)
        plt.savefig('waypoints_diff_y.png')
        max_y_diff = all_waypoints_diff[:,1].max().item()
        min_y_diff = all_waypoints_diff[:,1].min().item()
        print(f"Max y diff: {max_y_diff}, Min y diff: {min_y_diff}")
        plt.clf()
            
            

if __name__ == "__main__":
    test()