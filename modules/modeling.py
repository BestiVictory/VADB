from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import torch
from torch import nn
import torch.nn.functional as F

from modules.until_module import PreTrainedModel, AllGather, CrossEn
from modules.module_cross import CrossModel, CrossConfig, Transformer as TransformerClip

from modules.module_clip import CLIP, convert_weights
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

logger = logging.getLogger(__name__)
allgather = AllGather.apply

class CLIP4ClipPreTrainedModel(PreTrainedModel, nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, cross_config, *inputs, **kwargs):
        super(CLIP4ClipPreTrainedModel, self).__init__(cross_config)
        self.cross_config = cross_config
        self.clip = None
        self.clip_tag = None   #新增：tag编码器

        self.cross = None

    @classmethod
    
    def from_pretrained(cls, cross_model_name, state_dict=None, cache_dir=None, type_vocab_size=2, *inputs, **kwargs):

        task_config = None
        if "task_config" in kwargs.keys():
            task_config = kwargs["task_config"]
            if not hasattr(task_config, "local_rank"):
                task_config.__dict__["local_rank"] = 0
            elif task_config.local_rank == -1:
                task_config.local_rank = 0

        if state_dict is None: state_dict = {}
        pretrained_clip_name = "ViT-B/32"    #默认使用"ViT-B/32"作为预训练的CLIP模型名称，但如果task_config中有指定，则使用指定的名称。
        if hasattr(task_config, 'pretrained_clip_name'):
            pretrained_clip_name = task_config.pretrained_clip_name

        #加载clip预训练权重
        clip_state_dict = CLIP.get_config(pretrained_clip_name=pretrained_clip_name)


        # 为两个文本编码器准备权重
        for key, val in clip_state_dict.items():
            # 主文本编码器权重
            new_key = "clip." + key
            if new_key not in state_dict:
                state_dict[new_key] = val.clone()
            # 标签文本编码器权重
            new_key = "clip_tag." + key
            if new_key not in state_dict:
                state_dict[new_key] = val.clone()

        cross_config, _ = CrossConfig.get_config(cross_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)

        model = cls(cross_config, clip_state_dict, *inputs, **kwargs)

        ## ===> Initialization trick [HARD CODE]
        if model.linear_patch == "3d":
            contain_conv2 = False
            for key in state_dict.keys():
                if key.find("visual.conv2.weight") > -1:
                    contain_conv2 = True
                    break
            if contain_conv2 is False and hasattr(model.clip.visual, "conv2"):
                cp_weight = state_dict["clip.visual.conv1.weight"].clone()
                kernel_size = model.clip.visual.conv2.weight.size(2)
                conv2_size = model.clip.visual.conv2.weight.size()
                conv2_size = list(conv2_size)

                left_conv2_size = conv2_size.copy()
                right_conv2_size = conv2_size.copy()
                left_conv2_size[2] = (kernel_size - 1) // 2
                right_conv2_size[2] = kernel_size - 1 - left_conv2_size[2]

                left_zeros, right_zeros = None, None
                if left_conv2_size[2] > 0:
                    left_zeros = torch.zeros(*tuple(left_conv2_size), dtype=cp_weight.dtype, device=cp_weight.device)
                if right_conv2_size[2] > 0:
                    right_zeros = torch.zeros(*tuple(right_conv2_size), dtype=cp_weight.dtype, device=cp_weight.device)

                cat_list = []
                if left_zeros != None: cat_list.append(left_zeros)
                cat_list.append(cp_weight.unsqueeze(2))
                if right_zeros != None: cat_list.append(right_zeros)
                cp_weight = torch.cat(cat_list, dim=2)

                state_dict["clip.visual.conv2.weight"] = cp_weight
                state_dict["clip_tag.visual.conv2.weight"] = cp_weight.clone()  # 同时更新两个编码器

        ## <=== End of initialization trick

        if state_dict is not None:
            model = cls.init_preweight(model, state_dict, task_config=task_config)

        return model

def show_log(task_config, info):
    if task_config is None or task_config.local_rank == 0:
        logger.warning(info)

def update_attr(target_name, target_config, target_attr_name, source_config, source_attr_name, default_value=None):
    if hasattr(source_config, source_attr_name):
        if default_value is None or getattr(source_config, source_attr_name) != default_value:
            setattr(target_config, target_attr_name, getattr(source_config, source_attr_name))
            show_log(source_config, "Set {}.{}: {}.".format(target_name,
                                                            target_attr_name, getattr(target_config, target_attr_name)))
    return target_config

def check_attr(target_name, task_config):
    return hasattr(task_config, target_name) and task_config.__dict__[target_name]


class DynamicFusion(nn.Module):
    def __init__(self, hidden_size, text_bias=0.7):
        super().__init__()
        self.text_proj = nn.Linear(hidden_size, hidden_size)
        self.tag_proj = nn.Linear(hidden_size, hidden_size)
        
        # 注意力计算层
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),  # 修改为 hidden_size
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # 可学习的偏置参数
        self.text_bias = nn.Parameter(torch.tensor([text_bias]))
        self.tag_bias = nn.Parameter(torch.tensor([1 - text_bias]))
    
    def forward(self, text_feat, tag_feat):
        # 特征投影
        text_proj = self.text_proj(text_feat.squeeze(1))  # 去掉多余的维度 [16,512]
        tag_proj = self.tag_proj(tag_feat.squeeze(1))    # 去掉多余的维度 [16,512]
        
        # 计算注意力权重
        combined = torch.stack([text_proj, tag_proj], dim=1)  # [16, 2, 512]
        batch_size = combined.size(0)
        
        # 展平用于注意力计算
        flattened = combined.view(-1, combined.size(-1))  # [32, 512]
        raw_weights = self.attention(flattened)  # [32, 1]
        raw_weights = raw_weights.view(batch_size, 2, 1)  # [16, 2, 1]
        
        # 加入偏置
        biased_weights = raw_weights + torch.cat([self.text_bias, self.tag_bias], dim=0).view(1, 2, 1)
        attn_weights = F.softmax(biased_weights, dim=1)

        # 加权融合
        fused_feat = attn_weights[:, 0] * text_proj + attn_weights[:, 1] * tag_proj
        return fused_feat.unsqueeze(1)  # 恢复为 [16, 1, 512]

class CLIP4Clip(CLIP4ClipPreTrainedModel):
    def __init__(self, cross_config, clip_state_dict, task_config):
        super(CLIP4Clip, self).__init__(cross_config)

        self.task_config = task_config
        self.ignore_video_index = -1

        # 1.在 CLIP4Clip 的 __init__ 方法中调整
        cross_config.max_position_embeddings = self.task_config.max_words + self.task_config.max_frames

        # 2.重新初始化位置嵌入
        self.cross = CrossModel(cross_config)  # 重新加载交叉编码器

        #assert self.task_config.max_words + self.task_config.max_frames <= cross_config.max_position_embeddings

        self._stage_one = True
        self._stage_two = False

        show_log(task_config, "Stage-One:{}, Stage-Two:{}".format(self._stage_one, self._stage_two))

        self.loose_type = False
        if self._stage_one and check_attr('loose_type', self.task_config):
            self.loose_type = True
            show_log(task_config, "Test retrieval by loose type.")

        # CLIP Encoders: From OpenAI: CLIP [https://github.com/openai/CLIP] ===>
        vit = "visual.proj" in clip_state_dict
        assert vit
        if vit:
            vision_width = clip_state_dict["visual.conv1.weight"].shape[0]
            vision_layers = len(
                [k for k in clip_state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
            vision_patch_size = clip_state_dict["visual.conv1.weight"].shape[-1]
            grid_size = round((clip_state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
            image_resolution = vision_patch_size * grid_size
        else:
            counts: list = [len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"visual.layer{b}"))) for b in
                            [1, 2, 3, 4]]
            vision_layers = tuple(counts)
            vision_width = clip_state_dict["visual.layer1.0.conv1.weight"].shape[0]
            output_width = round((clip_state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
            vision_patch_size = None
            assert output_width ** 2 + 1 == clip_state_dict["visual.attnpool.positional_embedding"].shape[0]
            image_resolution = output_width * 32

        embed_dim = clip_state_dict["text_projection"].shape[1]  #文本编码器参数提取
        context_length = clip_state_dict["positional_embedding"].shape[0]
        vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
        transformer_width = clip_state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"transformer.resblocks")))

        show_log(task_config, "\t embed_dim: {}".format(embed_dim))
        show_log(task_config, "\t image_resolution: {}".format(image_resolution))
        show_log(task_config, "\t vision_layers: {}".format(vision_layers))
        show_log(task_config, "\t vision_width: {}".format(vision_width))
        show_log(task_config, "\t vision_patch_size: {}".format(vision_patch_size))
        show_log(task_config, "\t context_length: {}".format(context_length))
        show_log(task_config, "\t vocab_size: {}".format(vocab_size))
        show_log(task_config, "\t transformer_width: {}".format(transformer_width))
        show_log(task_config, "\t transformer_heads: {}".format(transformer_heads))
        show_log(task_config, "\t transformer_layers: {}".format(transformer_layers))

        self.linear_patch = '2d'
        if hasattr(task_config, "linear_patch"):
            self.linear_patch = task_config.linear_patch
            show_log(task_config, "\t\t linear_patch: {}".format(self.linear_patch))

        cut_top_layer = 0
        show_log(task_config, "\t cut_top_layer: {}".format(cut_top_layer))

        # 初始化编码器
        self.clip = CLIP(
            embed_dim,
            image_resolution, vision_layers-cut_top_layer, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers-cut_top_layer,
            linear_patch=self.linear_patch
        ).float()

        # 标签文本编码器 - 使用不同的初始化
        self.clip_tag = CLIP(
            embed_dim,
            image_resolution, vision_layers-cut_top_layer, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers-cut_top_layer,
            linear_patch=self.linear_patch
        ).float()
        
        # 共享视觉编码器权重
        self.clip_tag.visual.load_state_dict(self.clip.visual.state_dict())

        # 转换权重
        convert_weights(self.clip_tag)
        convert_weights(self.clip)

        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in clip_state_dict:
                del clip_state_dict[key]

        # <=== End of CLIP Encoders

        self.sim_header = 'meanP'
        if hasattr(task_config, "sim_header"):
            self.sim_header = task_config.sim_header
            show_log(task_config, "\t sim_header: {}".format(self.sim_header))

        cross_config.max_position_embeddings = context_length
        if self.loose_type is False:
            # Cross Encoder ===>
            cross_config = update_attr("cross_config", cross_config, "num_hidden_layers", self.task_config, "cross_num_hidden_layers")
            self.cross = CrossModel(cross_config)
            # <=== End of Cross Encoder
            self.similarity_dense = nn.Linear(cross_config.hidden_size, 1)

        self.loss_fct = CrossEn()

        self.apply(self.init_weights)

        self.dynamic_fusion = DynamicFusion(cross_config.hidden_size)    ######

    def forward(self, input_ids, token_type_ids, attention_mask, input_ids_tag, token_type_ids_tag, attention_mask_tag, video, video_mask=None):
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

        input_ids_tag = input_ids_tag.view(-1, input_ids_tag.shape[-1])
        token_type_ids_tag = token_type_ids_tag.view(-1, token_type_ids_tag.shape[-1])
        attention_mask_tag = attention_mask_tag.view(-1, attention_mask_tag.shape[-1])

        video_mask = video_mask.view(-1, video_mask.shape[-1])

        # T x 3 x H x W
        video = torch.as_tensor(video).float()
        b, pair, bs, ts, channel, h, w = video.shape
        video = video.view(b * pair * bs * ts, channel, h, w)
        video_frame = bs * ts

        sequence_output, sequence_output_tag, visual_output = self.get_sequence_visual_output(input_ids, token_type_ids, attention_mask,input_ids_tag, token_type_ids_tag,attention_mask_tag,video, video_mask, shaped=True, video_frame=video_frame)   #3.model 类的自定义方法调用栈

        fused_features = self.dynamic_fusion(sequence_output, sequence_output_tag)


        if self.training:
            loss = 0.
            sim_matrix, *_tmp = self.get_similarity_logits(fused_features, visual_output, attention_mask, video_mask, shaped=True, loose_type=self.loose_type)
            sim_loss1 = self.loss_fct(sim_matrix)
            sim_loss2 = self.loss_fct(sim_matrix.T)
            sim_loss = (sim_loss1 + sim_loss2) / 2
            loss += sim_loss

            return loss
        else:
            return None

    def get_sequence_output(self, input_ids, token_type_ids, attention_mask, shaped=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

        bs_pair = input_ids.size(0)
        sequence_hidden = self.clip.encode_text(input_ids).float()
        sequence_hidden = sequence_hidden.view(bs_pair, -1, sequence_hidden.size(-1))

        return sequence_hidden


    def get_sequence_output_tag(self, input_ids, token_type_ids, attention_mask, shaped=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

        bs_pair = input_ids.size(0)
        sequence_hidden_tag = self.clip_tag.encode_text(input_ids).float()
        sequence_hidden_tag = sequence_hidden_tag.view(bs_pair, -1, sequence_hidden_tag.size(-1))

        return sequence_hidden_tag

    def get_visual_output(self, video, video_mask, shaped=False, video_frame=-1):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        bs_pair = video_mask.size(0)
        visual_hidden = self.clip.encode_image(video, video_frame=video_frame).float()
        visual_hidden = visual_hidden.view(bs_pair, -1, visual_hidden.size(-1))

        return visual_hidden   #获取视频特征

    def get_sequence_visual_output(self, input_ids, token_type_ids, attention_mask, input_ids_tag, token_type_ids_tag, attention_mask_tag, video, video_mask, shaped=False, video_frame=-1):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

            input_ids_tag = input_ids_tag.view(-1, input_ids_tag.shape[-1])
            token_type_ids_tag = token_type_ids_tag.view(-1, token_type_ids_tag.shape[-1])
            attention_mask_tag = attention_mask_tag.view(-1, attention_mask_tag.shape[-1])


            video_mask = video_mask.view(-1, video_mask.shape[-1])

            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        sequence_output = self.get_sequence_output(input_ids, token_type_ids, attention_mask, shaped=True)
        sequence_output_tag = self.get_sequence_output_tag(input_ids_tag, token_type_ids_tag, attention_mask_tag, shaped=True)
        visual_output = self.get_visual_output(video, video_mask, shaped=True, video_frame=video_frame)

        return sequence_output, sequence_output_tag, visual_output   #同时获取文本和视频的特征表示

    def _mean_pooling_for_similarity_visual(self, visual_output, video_mask,):   #对视频特征序列进行掩码加权平均池化 
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        visual_output = visual_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum
        return video_out


    def _loose_similarity(self, fused_features, visual_output, attention_mask, video_mask, sim_header="meanP"):
        fused_features, visual_output = fused_features.contiguous(), visual_output.contiguous()

        print("before gather, ", torch.distributed.get_rank(), visual_output.shape)
        if self.training:
            visual_output = allgather(visual_output, self.task_config)
            video_mask = allgather(video_mask, self.task_config)
            fused_features = allgather(fused_features, self.task_config)
            torch.distributed.barrier()
        print("after gather, ", torch.distributed.get_rank(), visual_output.shape)

        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        visual_output = self._mean_pooling_for_similarity_visual(visual_output, video_mask)
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)

        fused_features = fused_features.squeeze(1)
        fused_features = fused_features / fused_features.norm(dim=-1, keepdim=True)

        logit_scale = self.clip.logit_scale.exp()
        retrieve_logits = logit_scale * torch.matmul(fused_features, visual_output.t())
        return retrieve_logits

    def get_similarity_logits(self, fused_features, visual_output, attention_mask, video_mask, shaped=False, loose_type=False):
        if shaped is False:
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

        contrastive_direction = ()
        retrieve_logits = self._loose_similarity(fused_features, visual_output, attention_mask, video_mask, sim_header=self.sim_header)
        return retrieve_logits, contrastive_direction