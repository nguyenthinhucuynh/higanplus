import torch


def apply_vertical_stripe_mask(imgs, img_lens, mask_ratio_range=(0.05, 0.15), stripe_width_range=(1, 4)):
    """
    Apply vertical stripe masking to generated images.

    Args:
        imgs: [B, C, H, W] image tensor
        img_lens: [B] length tensor
        mask_ratio_range: tuple of (min, max) ratio of image to mask
        stripe_width_range: tuple of (min, max) stripe width in pixels

    Returns:
        masked_imgs: [B, C, H, W] masked image tensor
    """
    B, C, H, W = imgs.shape
    masked_imgs = imgs.clone()

    for i in range(B):
        valid_len = int(img_lens[i].item())
        if valid_len <= 0:
            continue

        mask_ratio = torch.rand(1).item() * (mask_ratio_range[1] - mask_ratio_range[0]) + mask_ratio_range[0]
        num_pixels_to_mask = int(valid_len * mask_ratio)

        stripe_width = torch.randint(stripe_width_range[0], stripe_width_range[1] + 1, (1,)).item()

        num_stripes = max(1, num_pixels_to_mask // stripe_width)
        for _ in range(num_stripes):
            start_pos = torch.randint(0, max(1, valid_len - stripe_width), (1,)).item()
            end_pos = min(start_pos + stripe_width, valid_len)
            masked_imgs[i, :, :, start_pos:end_pos] = -1

    return masked_imgs


def apply_horizontal_stripe_mask(imgs, img_lens, mask_ratio_range=(0.05, 0.15), stripe_height_range=(1, 4)):
    """
    Apply horizontal stripe masking to generated images.

    Args:
        imgs: [B, C, H, W] image tensor
        img_lens: [B] length tensor
        mask_ratio_range: tuple of (min, max) ratio of image height to mask
        stripe_height_range: tuple of (min, max) stripe height in pixels

    Returns:
        masked_imgs: [B, C, H, W] masked image tensor
    """
    B, C, H, W = imgs.shape
    masked_imgs = imgs.clone()

    for i in range(B):
        mask_ratio = torch.rand(1).item() * (mask_ratio_range[1] - mask_ratio_range[0]) + mask_ratio_range[0]
        num_pixels_to_mask = int(H * mask_ratio)

        stripe_height = torch.randint(stripe_height_range[0], stripe_height_range[1] + 1, (1,)).item()

        num_stripes = max(1, num_pixels_to_mask // stripe_height)
        valid_len = int(img_lens[i].item())

        for _ in range(num_stripes):
            start_pos = torch.randint(0, max(1, H - stripe_height), (1,)).item()
            end_pos = min(start_pos + stripe_height, H)
            masked_imgs[i, :, start_pos:end_pos, :valid_len] = -1

    return masked_imgs


def apply_combined_stripe_mask(imgs, img_lens, mask_ratio_range=(0.05, 0.15),
                                stripe_width_range=(1, 4), stripe_height_range=(1, 4)):
    """
    Apply both vertical and horizontal stripe masking to generated images.

    Args:
        imgs: [B, C, H, W] image tensor
        img_lens: [B] length tensor
        mask_ratio_range: tuple of (min, max) ratio to mask
        stripe_width_range: tuple of (min, max) vertical stripe width
        stripe_height_range: tuple of (min, max) horizontal stripe height

    Returns:
        masked_imgs: [B, C, H, W] masked image tensor
    """
    masked_imgs = apply_vertical_stripe_mask(imgs, img_lens, mask_ratio_range, stripe_width_range)
    masked_imgs = apply_horizontal_stripe_mask(masked_imgs, img_lens, mask_ratio_range, stripe_height_range)
    return masked_imgs
