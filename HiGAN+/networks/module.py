import numpy as np
import torch
from torch import nn
import functools
from networks.block import Conv2dBlock, ActFirstResBlock, DeepBLSTM, DeepGRU, DeepLSTM, Identity
from networks.utils import _len2mask, init_weights


class StyleBackbone(nn.Module):
    def __init__(self, resolution=16, max_dim=256, in_channel=1, init='N02', dropout=0.0, norm='bn'):
        super(StyleBackbone, self).__init__()
        self.reduce_len_scale = 16
        nf = resolution
        cnn_f = [nn.ConstantPad2d(2, -1),
                 Conv2dBlock(in_channel, nf, 5, 2, 0,
                             norm='none',
                             activation='none')]
        for i in range(2):
            nf_out = min([int(nf * 2), max_dim])
            cnn_f += [ActFirstResBlock(nf, nf, None, 'relu', norm, 'zero', dropout=dropout / 2)]
            cnn_f += [nn.ZeroPad2d((1, 1, 0, 0))]
            cnn_f += [ActFirstResBlock(nf, nf_out, None, 'relu', norm, 'zero', dropout=dropout / 2)]
            cnn_f += [nn.ZeroPad2d(1)]
            cnn_f += [nn.MaxPool2d(kernel_size=3, stride=2)]
            nf = min([nf_out, max_dim])

        df = nf
        for i in range(2):
            df_out = min([int(df * 2), max_dim])
            cnn_f += [ActFirstResBlock(df, df, None, 'relu', norm, 'zero', dropout=dropout)]
            cnn_f += [ActFirstResBlock(df, df_out, None, 'relu', norm, 'zero', dropout=dropout)]
            if i < 1:
                cnn_f += [nn.MaxPool2d(kernel_size=3, stride=2)]
            else:
                cnn_f += [nn.ZeroPad2d((1, 1, 0, 0))]
            df = min([df_out, max_dim])
        self.cnn_backbone = nn.Sequential(*cnn_f)
        self.layer_name_mapping = {
            '9': "feat2",
            '13': "feat3",
            '16': "feat4",
        }

        self.cnn_ctc = nn.Sequential(
            nn.ReLU(),
            Conv2dBlock(df, df, 3, 1, 0,
                        norm=norm,
                        activation='relu')
        )
        if init != 'none':
            init_weights(self, init)

    def forward(self, x, ret_feats=False):
        feats = []
        for name, layer in self.cnn_backbone._modules.items():
            x = layer(x)
            if ret_feats and name in self.layer_name_mapping:
                feats.append(x)

        out = self.cnn_ctc(x).squeeze(-2)

        return out, feats


class VerticalMaskingHead(nn.Module):
    """
    Vertical style enhancing head via column-wise masking.
    Enhances style learning in the vertical direction (e.g., vertical alignment patterns).
    """
    def __init__(self, in_dim=256, style_dim=32, mask_ratio=0.3, init='N02'):
        super(VerticalMaskingHead, self).__init__()
        self.mask_ratio = mask_ratio

        self.vertical_encoder = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(),
            nn.Linear(in_dim, style_dim),
        )

        if init != 'none':
            init_weights(self, init)

    def apply_column_mask(self, feat, img_len_mask):
        """
        Apply column-wise masking to features.
        Args:
            feat: [B, C, W] feature tensor
            img_len_mask: [B, 1, W] length mask
        Returns:
            masked_feat: [B, C, W] masked feature tensor
            mask: [B, 1, W] binary mask
        """
        B, C, W = feat.shape
        device = feat.device

        # Create column-wise mask
        num_mask = int(W * self.mask_ratio)
        mask = torch.ones(B, 1, W, device=device)

        for i in range(B):
            # Randomly select columns to mask
            valid_len = int(img_len_mask[i].sum().item())
            if valid_len > 0:
                mask_indices = torch.randperm(valid_len, device=device)[:num_mask]
                mask[i, :, mask_indices] = 0

        masked_feat = feat * mask
        return masked_feat, mask

    def forward(self, feat, img_len, img_len_mask):
        """
        Forward pass with column-wise masking.
        Args:
            feat: [B, C, W] feature tensor
            img_len: [B] length tensor
            img_len_mask: [B, 1, W] length mask
        Returns:
            vertical_style: [B, style_dim] vertical style vector
        """
        # Apply column-wise masking
        masked_feat, mask = self.apply_column_mask(feat, img_len_mask)

        # Aggregate masked features
        masked_len = (img_len_mask * mask).sum(dim=-1).squeeze(1)
        vertical_feat = (masked_feat * img_len_mask).sum(dim=-1) / (masked_len.unsqueeze(1).float() + 1e-8)

        # Encode to style vector
        vertical_style = self.vertical_encoder(vertical_feat)

        return vertical_style


class HorizontalMaskingHead(nn.Module):
    """
    Horizontal style enhancing head via row-wise masking.
    Enhances style learning in the horizontal direction (e.g., word and character spacing).
    """
    def __init__(self, in_dim=256, style_dim=32, mask_ratio=0.3, init='N02'):
        super(HorizontalMaskingHead, self).__init__()
        self.mask_ratio = mask_ratio

        self.horizontal_encoder = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(),
            nn.Linear(in_dim, style_dim),
        )

        if init != 'none':
            init_weights(self, init)

    def apply_row_mask(self, feat):
        """
        Apply row-wise masking to features.
        Args:
            feat: [B, C, W] feature tensor
        Returns:
            masked_feat: [B, C, W] masked feature tensor
            mask: [B, C, 1] binary mask
        """
        B, C, W = feat.shape
        device = feat.device

        # Create row-wise mask
        num_mask = int(C * self.mask_ratio)
        mask = torch.ones(B, C, 1, device=device)

        for i in range(B):
            # Randomly select rows (channels) to mask
            mask_indices = torch.randperm(C, device=device)[:num_mask]
            mask[i, mask_indices, :] = 0

        masked_feat = feat * mask
        return masked_feat, mask

    def forward(self, feat, img_len, img_len_mask):
        """
        Forward pass with row-wise masking.
        Args:
            feat: [B, C, W] feature tensor
            img_len: [B] length tensor
            img_len_mask: [B, 1, W] length mask
        Returns:
            horizontal_style: [B, style_dim] horizontal style vector
        """
        # Apply row-wise masking
        masked_feat, mask = self.apply_row_mask(feat)

        # Aggregate masked features
        horizontal_feat = (masked_feat * img_len_mask).sum(dim=-1) / (img_len.unsqueeze(1).float() + 1e-8)

        # Encode to style vector
        horizontal_style = self.horizontal_encoder(horizontal_feat)

        return horizontal_style


class StyleEncoder(nn.Module):
    def __init__(self, style_dim=32, in_dim=256, init='N02', use_masking=True, mask_ratio=0.3):
        super(StyleEncoder, self).__init__()
        self.style_dim = style_dim
        self.use_masking = use_masking

        ######################################
        # Construct StyleEncoder
        ######################################
        self.linear_style = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(),
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(),
        )

        self.mu = nn.Linear(in_dim, style_dim)
        self.logvar = nn.Linear(in_dim, style_dim)

        # Masking heads for enhanced style learning
        if self.use_masking:
            self.vertical_head = VerticalMaskingHead(in_dim, style_dim, mask_ratio, init)
            self.horizontal_head = HorizontalMaskingHead(in_dim, style_dim, mask_ratio, init)

            # Fusion layer to combine base style with masked styles
            self.style_fusion = nn.Sequential(
                nn.Linear(style_dim * 3, style_dim * 2),
                nn.LeakyReLU(),
                nn.Linear(style_dim * 2, style_dim),
            )

        if init != 'none':
            init_weights(self, init)

    def forward(self, img, img_len, cnn_backbone=None, ret_feats=False, vae_mode=False, ret_masked_styles=False):
        feat, all_feats = cnn_backbone(img, ret_feats)
        img_len = img_len // cnn_backbone.reduce_len_scale
        img_len_mask = _len2mask(img_len, feat.size(-1)).unsqueeze(1).float().detach()

        # Base style encoding
        style = (feat * img_len_mask).sum(dim=-1) / (img_len.unsqueeze(1).float() + 1e-8)
        style = self.linear_style(style)
        mu = self.mu(style)

        # Enhanced style with masking heads
        masked_styles = None
        if self.use_masking and self.training and ret_masked_styles:
            vertical_style = self.vertical_head(feat, img_len, img_len_mask)
            horizontal_style = self.horizontal_head(feat, img_len, img_len_mask)
            masked_styles = (vertical_style, horizontal_style)

            # Fuse base style with masked styles
            combined_style = torch.cat([mu, vertical_style, horizontal_style], dim=-1)
            mu = self.style_fusion(combined_style)
        elif self.use_masking and self.training:
            # Apply masking but don't return masked styles
            vertical_style = self.vertical_head(feat, img_len, img_len_mask)
            horizontal_style = self.horizontal_head(feat, img_len, img_len_mask)

            # Fuse base style with masked styles
            combined_style = torch.cat([mu, vertical_style, horizontal_style], dim=-1)
            mu = self.style_fusion(combined_style)

        if vae_mode:
            logvar = self.logvar(style)
            style = self.reparameterize(mu, logvar)
            final_style = (style, mu, logvar)
        else:
            final_style = mu

        # Return based on flags
        if ret_feats and ret_masked_styles and masked_styles is not None:
            return final_style, all_feats, masked_styles
        elif ret_feats:
            return final_style, all_feats
        elif ret_masked_styles and masked_styles is not None:
            return final_style, masked_styles
        else:
            return final_style

    @staticmethod
    def reparameterize(mu, logvar):
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu


class WriterIdentifier(nn.Module):
    def __init__(self, n_writer=372, in_dim=256, init='N02'):
        super(WriterIdentifier, self).__init__()
        self.reduce_len_scale = 32

        ######################################
        # Construct WriterIdentifier
        ######################################

        self.linear_wid = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(),
            nn.Linear(in_dim, n_writer),
        )

        if init != 'none':
            init_weights(self, init)

    def forward(self, img, img_len, cnn_backbone, ret_feats=False):
        feat, all_feats = cnn_backbone(img, ret_feats)
        img_len = img_len // cnn_backbone.reduce_len_scale
        img_len_mask = _len2mask(img_len, feat.size(-1)).unsqueeze(1).float().detach()
        wid_feat = (feat * img_len_mask).sum(dim=-1) / (img_len.unsqueeze(1).float() + 1e-8)
        wid_logits = self.linear_wid(wid_feat)
        if ret_feats:
            return wid_logits, all_feats
        else:
            return wid_logits

    def return_feat(self, img, img_len):
        feat = self.cnn_backbone(img)
        img_len = img_len // self.reduce_len_scale
        out_w = self.cnn_wid(feat).squeeze(-2)
        img_len_mask = _len2mask(img_len, out_w.size(-1)).unsqueeze(1).float().detach()
        wid_feat = (out_w * img_len_mask).sum(dim=-1) / (img_len.unsqueeze(1).float() + 1e-8)
        for j in range(2):
            wid_feat = self.linear_wid[j](wid_feat)
        return wid_feat


class Recognizer(nn.Module):
    # resolution: 32  max_dim: 512  in_channel: 1  norm: 'none'  init: 'N02'  dropout: 0.  n_class: 72  rnn_depth: 0
    def __init__(self, n_class, resolution=16, max_dim=256, in_channel=1, norm='none',
                 init='none', rnn_depth=1, dropout=0.0, bidirectional=True):
        super(Recognizer, self).__init__()
        self.len_scale = 16
        self.use_rnn = rnn_depth > 0
        self.bidirectional = bidirectional

        ######################################
        # Construct Backbone
        ######################################
        nf = resolution
        cnn_f = [nn.ConstantPad2d(2, -1),
                 Conv2dBlock(in_channel, nf, 5, 2, 0,
                             norm='none',
                             activation='none')]
        for i in range(2):
            nf_out = min([int(nf * 2), max_dim])
            cnn_f += [ActFirstResBlock(nf, nf, None, 'relu', norm, 'zero', dropout=dropout / 2)]
            cnn_f += [nn.ZeroPad2d((1, 1, 0, 0))]
            cnn_f += [ActFirstResBlock(nf, nf_out, None, 'relu', norm, 'zero', dropout=dropout / 2)]
            cnn_f += [nn.ZeroPad2d(1)]
            cnn_f += [nn.MaxPool2d(kernel_size=3, stride=2)]
            nf = min([nf_out, max_dim])

        df = nf
        for i in range(2):
            df_out = min([int(df * 2), max_dim])
            cnn_f += [ActFirstResBlock(df, df, None, 'relu', norm, 'zero', dropout=dropout)]
            cnn_f += [ActFirstResBlock(df, df_out, None, 'relu', norm, 'zero', dropout=dropout)]
            if i < 1:
                cnn_f += [nn.MaxPool2d(kernel_size=3, stride=2)]
            else:
                cnn_f += [nn.ZeroPad2d((1, 1, 0, 0))]
            df = min([df_out, max_dim])

        ######################################
        # Construct Classifier
        ######################################
        cnn_c = [nn.ReLU(),
                 Conv2dBlock(df, df, 3, 1, 0,
                             norm=norm,
                             activation='relu')]

        self.cnn_backbone = nn.Sequential(*cnn_f)
        self.cnn_ctc = nn.Sequential(*cnn_c)
        if self.use_rnn:
            if bidirectional:
                self.rnn_ctc = DeepBLSTM(df, df, rnn_depth, bidirectional=True)
            else:
                self.rnn_ctc = DeepLSTM(df, df, rnn_depth)
        self.ctc_cls = nn.Linear(df, n_class)

        if init != 'none':
            init_weights(self, init)

    def forward(self, x, x_len=None):
        cnn_feat = self.cnn_backbone(x)
        cnn_feat2 = self.cnn_ctc(cnn_feat)
        ctc_feat = cnn_feat2.squeeze(-2).transpose(1, 2)
        if self.use_rnn:
            if self.bidirectional:
                ctc_len = x_len // (self.len_scale + 1e-8)
            else:
                ctc_len = None
            ctc_feat = self.rnn_ctc(ctc_feat, ctc_len.cpu())
        logits = self.ctc_cls(ctc_feat)
        if self.training:
            logits = logits.transpose(0, 1).log_softmax(2)
            logits.requires_grad_(True)
        return logits

    def frozen_bn(self):
        def fix_bn(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()
        self.apply(fix_bn)

