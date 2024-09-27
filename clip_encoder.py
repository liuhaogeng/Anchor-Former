import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        selected_query = None
        if isinstance(image_forward_outs.attentions, tuple) :
            image_features = image_forward_outs.hidden_states[self.select_layer]

            # code for vit cls token selection
            attention = image_forward_outs.attentions[self.select_layer]
            num_count = 256
            per_head_num = int(num_count / attention.shape[1])
            for ind in range(attention.shape[0]):
                max_indices = []
                for ind_y in range(attention.shape[1]):
                    tmp_attn = attention[ind, ind_y, 0, 1:]
                    tmp_ind_sorted = (torch.argsort(tmp_attn) + 1).detach().cpu().tolist()
                    tmp_res = set(max_indices + tmp_ind_sorted[-per_head_num:] + [0])
                    count = 1
                    while len(tmp_res) < (ind_y * per_head_num + per_head_num + 1) :
                        tmp_res = set(max_indices + tmp_ind_sorted[-per_head_num-count:] + [0])
                        count += 1
                    max_indices = sorted(list(tmp_res))
                if torch.is_tensor(selected_query):
                    selected_query = torch.cat((selected_query, image_features[ind, max_indices, :].unsqueeze(0)), dim=0)
                else:
                    selected_query = image_features[ind, max_indices, :].unsqueeze(0)

            # mean pooling
            # bs, _, d = image_features.size()
            # tmp_query = image_features[:, 1:, :].reshape(bs, 24, 24, d)
            # selected_query = image_features[:, 0, :].unsqueeze(1)
            # for ind_x in range(12):
            #     for ind_y in range(12):
            #         st_x = ind_x * 2
            #         ed_x = ind_x * 2 + 2
            #         st_y = ind_y * 2
            #         ed_y = ind_y * 2 + 2
            #         tmp_emb = tmp_query[:, st_x:ed_x, st_y:ed_y, :].reshape(bs, -1 ,d)
            #         tmp_emb = torch.mean(tmp_emb, dim=1, keepdim=True)
            #         selected_query = torch.cat((selected_query, tmp_emb), dim=1)
            # print(selected_query.shape)
            
            return image_features, selected_query
        else:
            image_features = image_forward_outs.hidden_states[self.select_layer]
            if self.select_feature == 'patch':
                image_features = image_features[:, 1:]
            elif self.select_feature == 'cls_patch':
                image_features = image_features
            else:
                raise ValueError(f'Unexpected select feature: {self.select_feature}')
            return image_features, selected_query

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True, output_attentions=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True, output_attentions=True)
            image_features, selected_query = self.feature_select(image_forward_outs)
            image_features = image_features.to(images.dtype)
            selected_query = selected_query.to(images.dtype)
        return image_features, selected_query

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2



class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)

        self.s2_scales = getattr(args, 's2_scales', '336,672,1008')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError('Package s2wrapper not found! Please install by running: \npip install git+https://github.com/bfshi/scaling_on_scales.git')
        self.multiscale_forward = multiscale_forward

        # change resize/crop size in preprocessing to the largest image size in s2_scale
        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.image_processor.size['shortest_edge'] = self.s2_image_size
            self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.image_processor.size['shortest_edge'] = self.s2_image_size
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

        self.is_loaded = True

    @torch.no_grad()
    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(self.forward_feature, image.unsqueeze(0), img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
                image_features.append(image_feature)
        else:
            image_features = self.multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size)

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)
