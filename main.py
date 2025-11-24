import numpy as np
import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from forwardImagingPoisson import ForwardImaging
from canopyPlots import createCHM
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import math
import warnings
warnings.filterwarnings("ignore")

# set random seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Constants
RESOLUTION = 2
FACTOR = 5
SAMPLING_TIMESTEPS = 10
IMAGE_SIZE = 96 // RESOLUTION
DEVICE = 'cuda'

# Magic numbers as constants
LR_MULTIPLIER = 2
EPSILON = 1e-16
RECON_THRESHOLD = 0.05

cube_tag = f"timesteps_{SAMPLING_TIMESTEPS}"

def load_and_preprocess_data():
    """
    Load and preprocess input and ground truth data.

    Args:
        reduce_memory (bool): If True, reduce data size to avoid memory issues.

    Returns:
        tuple: Preprocessed input_data and ground_truth tensors.
    """
    input_data = torch.from_numpy(np.load(f'TestCube/entrada3.npy')).float().to(DEVICE)
    ground_truth = torch.from_numpy(np.swapaxes(np.load('TestCube/gt2.npy'),-1,-2)).float().to(DEVICE)

    # Adjust dimensions based on factor
    input_data = input_data[:, :32 * FACTOR, :16 * FACTOR]
    ground_truth = ground_truth[:, :(96 // RESOLUTION) * FACTOR, :(96 // RESOLUTION) * FACTOR]

    # Normalize data
    input_data = input_data / (input_data.max(dim=0)[0] + EPSILON)
    ground_truth = ground_truth / (ground_truth.max(dim=0)[0] + EPSILON)

    print(input_data.shape, ground_truth.shape)

    return input_data, ground_truth

def create_mask(input_shape, ratio, mask_type='blue_noise'):
    """
    Create a mask for sampling with different options.

    Args:
        input_shape (tuple): Shape of the input data tensor.
        ratio (float): Fraction of elements to sample.
        mask_type (str): Type of mask. Options are:
            - 'random': Creates a random mask.
            - 'blue_noise': Loads a blue noise image (Mblue.tiff) and thresholds it.
            - 'bayer': Constructs a Bayer pattern mask.
            
    Returns:
        torch.Tensor: Mask tensor.
    """
    assert ratio <= 1, "Ratio must be less than or equal to 1."
    mask_size = (round(input_shape[-2]), input_shape[-1])
    
    if mask_type == 'random':
        return torch.rand(mask_size, device=DEVICE) < ratio
    
    elif mask_type == 'blue_noise':
        blue_noise_img = Image.open("Mblue.tiff").convert('L')
        blue_noise = np.array(blue_noise_img, dtype=np.float32) / 255.0
        blue_noise = blue_noise[:mask_size[0], :mask_size[1]]
        return torch.tensor(blue_noise, device=DEVICE) < ratio

    elif mask_type == 'bayer':
        bayer = np.array([[0,  8,  2, 10],
                          [12, 4, 14,  6],
                          [3, 11,  1,  9],
                          [15, 7, 13,  5]], dtype=np.float32)
        bayer = bayer / 16.0
        mask_rows, mask_cols = mask_size
        tile_rows = int(math.ceil(mask_rows / bayer.shape[0]))
        tile_cols = int(math.ceil(mask_cols / bayer.shape[1]))
        tiled = np.tile(bayer, (tile_rows, tile_cols))[:mask_rows, :mask_cols]
        return torch.tensor(tiled, device=DEVICE) < ratio

    else:
        raise ValueError("Invalid mask_type. Choose among 'random', 'blue_noise', or 'bayer'.")

def create_sampling_matrix(mask: torch.Tensor) -> torch.Tensor:
    """
    Create a sampling matrix from a binary mask.
    
    Args:
        mask (torch.Tensor): Binary mask tensor of shape (H, W).

    Returns:
        torch.Tensor: Sampling matrix of shape (H*W, k).
    """
    flat_mask = mask.view(-1)
    ones_indices = torch.nonzero(flat_mask == 1, as_tuple=False).squeeze()
    k = ones_indices.numel()
    sampling_matrix = torch.zeros(flat_mask.numel(), k, device=mask.device, dtype=mask.dtype)
    
    for j, idx in enumerate(ones_indices):
        sampling_matrix[idx, j] = 1
        
    return sampling_matrix.float()

def plot_results(input_image: torch.Tensor,
                 sample: torch.Tensor,
                 gt_image: torch.Tensor) -> None:
    """
    Plot the CHM, DTM, and profile comparisons for input, reconstruction, and ground truth.

    Args:
        input_image (torch.Tensor): Input image tensor.
        sample (torch.Tensor): Reconstructed sample tensor.
        gt_image (torch.Tensor): Ground truth image tensor.
    """
    # Convert tensors to NumPy arrays.
    input_np = input_image.cpu().numpy()
    sample_np = sample#.cpu().numpy()
    gt_np = gt_image.cpu().numpy()

    # Create CHM, DTM, and hillshade images.
    chm_input, dtm_input, hillshade_input, _ = createCHM(input_np, porcentaje=0.95)
    chm_recon, dtm_recon, hillshade_recon, _ = createCHM(sample_np, porcentaje=0.95)
    chm_gt, dtm_gt, hillshade_gt, _ = createCHM(gt_np, porcentaje=0.95)

    chm_input = chm_input*0.5
    chm_recon = chm_recon*0.5
    chm_gt = chm_gt*0.5

    dtm_input = dtm_input*0.5
    dtm_recon = dtm_recon*0.5
    dtm_gt = dtm_gt*0.5

    # Create a 3x3 subplot.
    fig, axs = plt.subplots(3, 4, figsize=(15, 15))

    # CHM plots.
    axs[0, 0].imshow(chm_input, cmap='viridis')
    axs[0, 0].set_aspect(0.5)
    axs[0, 0].set_title('CHM Input')

    axs[0, 1].imshow(chm_recon, cmap='viridis', vmin=chm_gt.min(),
                      vmax=chm_gt.max())
    axs[0, 1].set_title('CHM Reconstruction')

    axs[0, 2].imshow(chm_gt, cmap='viridis')
    axs[0, 2].set_title('CHM Ground Truth')

    # DTM plots with hillshade overlay.
    axs[1, 0].imshow(dtm_input, cmap='copper')
    axs[1, 0].imshow(hillshade_input, cmap='Grays', alpha=0.35)
    axs[1, 0].set_aspect(0.5)
    axs[1, 0].set_title('DTM Input')

    axs[1, 1].imshow(dtm_recon, cmap='copper', vmin=dtm_gt.min(),
                      vmax=dtm_gt.max())
    axs[1, 1].imshow(hillshade_recon, cmap='Grays', alpha=0.35)
    axs[1, 1].set_title('DTM Reconstruction')

    axs[1, 2].imshow(dtm_gt, cmap='copper')
    axs[1, 2].imshow(hillshade_gt, cmap='Grays', alpha=0.35)
    axs[1, 2].set_title('DTM Ground Truth')

    # Profile plots.
    profile_index_input = 3 * input_np.shape[1] // 4
    profile_index_gt = 3 * gt_np.shape[1] // 4

    axs[2, 0].imshow(input_np[::-1, profile_index_input, :],
                      cmap='gray_r', interpolation='nearest')
    axs[2, 0].set_aspect(1 / 3)
    axs[2, 0].set_title('Profile Input')

    axs[2, 1].imshow(sample_np[::-1, profile_index_gt, :],
                      cmap='gray_r', interpolation='nearest')
    axs[2, 1].set_title('Profile Reconstruction')

    axs[2, 2].imshow(gt_np[::-1, profile_index_gt, :],
                      cmap='gray_r', interpolation='nearest')
    axs[2, 2].set_title('Profile Ground Truth')

    # Error plots.
    error_chm = np.abs(chm_recon - chm_gt)
    error_dtm = np.abs(dtm_recon - dtm_gt)
    error_profile = np.abs(sample_np[::-1, profile_index_gt, :] - gt_np[::-1, profile_index_gt, :])

    im0 = axs[0, 3].imshow(error_chm, cmap='turbo')
    axs[0, 3].set_title('CHM Error')
    fig.colorbar(im0, ax=axs[0, 3])

    im1 = axs[1, 3].imshow(error_dtm, cmap='turbo')
    axs[1, 3].set_title('DTM Error')
    fig.colorbar(im1, ax=axs[1, 3])

    im2 = axs[2, 3].imshow(error_profile, cmap='turbo')
    axs[2, 3].set_title('Profile Error')
    fig.colorbar(im2, ax=axs[2, 3])

    #plt.show()
    plt.savefig(f"results/reconstruction_{cube_tag}.png", dpi=200)
    plt.close()
    

    # Compute metrics (SSIM and PSNR) for CHM and DTM.
    from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
    ssim_chm = ssim(chm_gt, chm_recon, data_range=chm_gt.max() - chm_gt.min(),win_size=15)
    psnr_chm = psnr(chm_gt, chm_recon, data_range=chm_gt.max() - chm_gt.min())
    ssim_dtm = ssim(dtm_gt, dtm_recon, data_range=dtm_gt.max() - dtm_gt.min(),win_size=15)
    psnr_dtm = psnr(dtm_gt, dtm_recon, data_range=dtm_gt.max() - dtm_gt.min())
    print(f"SSIM CHM: {ssim_chm:.4f}, PSNR CHM: {psnr_chm:.4f}")
    print(f"SSIM DTM: {ssim_dtm:.4f}, PSNR DTM: {psnr_dtm:.4f}")

    # compute lpips metric
    import lpips
    lpips_fn = lpips.LPIPS(net='vgg')
    lpips_chm = lpips_fn.forward(torch.tensor(chm_gt).float().unsqueeze(0).unsqueeze(0),
                                  torch.tensor(chm_recon).float().unsqueeze(0).unsqueeze(0)).item()
    lpips_dtm = lpips_fn.forward(torch.tensor(dtm_gt).float().unsqueeze(0).unsqueeze(0),
                                    torch.tensor(dtm_recon).float().unsqueeze(0).unsqueeze(0)).item()
    print(f"LPIPS CHM: {lpips_chm:.4f}, LPIPS DTM: {lpips_dtm:.4f}")

    # compute MSE and MAE for the whole tensor
    mse = torch.mean((torch.tensor(gt_np) - torch.tensor(sample_np)) ** 2).item()
    mae = torch.mean(torch.abs(torch.tensor(gt_np) - torch.tensor(sample_np))).item()
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}")

def most_likely_Y_batch(X, n=20):
    """
    Compute the most likely count vectors Y for a batch of multinomial distributions.
    
    Args:
        X (torch.Tensor): Tensor of shape (batch_size, num_dist, k), where each X[i,j,:] is a probability vector summing to 1
        n (int): Number of trials, default is 20
    
    Returns:
        torch.Tensor: Tensor Y of shape (batch_size, num_dist, k) with integer counts summing to n for each distribution
    """
    X = X.transpose(0, -1)  # Transpose to shape (k, batch_size, num_dist)
    batch_size, num_dist, k = X.shape  # e.g., 128, 32, 16
    Y = torch.zeros(batch_size, num_dist, k, dtype=torch.int64)
    
    for _ in range(n):
        scores = X / (Y + 1)  # Shape: (128, 32, 16)
        idx = torch.argmax(scores, dim=-1)  # Shape: (128, 32)
        batch_idx = torch.arange(batch_size).view(-1, 1).expand(-1, num_dist)  # Shape: (128, 32)
        dist_idx = torch.arange(num_dist).view(1, -1).expand(batch_size, -1)  # Shape: (128, 32)
        Y[batch_idx, dist_idx, idx] += 1  # Increment Y at specified indices
    
    return Y.transpose(0,-1)

def main():
    """Main function to execute the diffusion model inference and visualization for multiple simulations."""

    # Initialize forward imaging
    forward_imaging = ForwardImaging(RESOLUTION, device=DEVICE, photons=20)

    # Load and preprocess data
    input_data, ground_truth = load_and_preprocess_data()

    print(input_data.min(), input_data.max(), input_data.shape)

    # Define U-Net model
    model = Unet(
        dim=128,
        dim_mults=(8, 16, 16, 16),
        flash_attn=True,
        channels=128
    )

    # Set up Gaussian Diffusion
    diffusion = GaussianDiffusion(
        model,
        image_size=IMAGE_SIZE,
        timesteps=1000,
        sampling_timesteps=SAMPLING_TIMESTEPS
    )

    # Initialize Trainer
    trainer = Trainer(
        diffusion,
        train_batch_size=8,
        train_lr=8e-5,
        train_num_steps=700000,
        gradient_accumulate_every=8,
        ema_decay=0.995,
        amp=True,
        resolution=RESOLUTION
    )

    # Load pre-trained model
    trainer.load(0)
    trainer.ema.ema_model.eval()

    ################## parameters for ddim
    times = torch.linspace(-1, 1000 - 1, steps = SAMPLING_TIMESTEPS + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
    times = list(reversed(times.int().tolist()))
    time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
    eta = 1.0
    ##################

    # Compute y
    y = input_data

    # mask = torch.rand(y.shape[-2], y.shape[-1], device=DEVICE) < 0.25
    mask = create_mask(y.shape, ratio=1, mask_type='blue_noise')
    pattern = create_sampling_matrix(mask).T.cuda()
    mask = mask.expand_as(y).float()

    # Initialize output
    output = torch.randn_like(ground_truth.unsqueeze(0).float(), device=DEVICE)

    # Iterative refinement loop
    pbar = tqdm(time_pairs, total=SAMPLING_TIMESTEPS)
    for time, time_next in pbar:
        output = output.requires_grad_()
        # start "p_sample"
        time_cond = torch.full((1,), time, device = 'cuda', dtype = torch.long)
        pred_noise, x_start, *_ = trainer.ema.ema_model.model_predictions(output, time_cond, x_self_cond = None, clip_x_start = True, rederive_pred_noise = True)

        if time_next < 0:
            output = x_start
            continue

        alpha = trainer.model.alphas_cumprod[time]
        alpha_next = trainer.model.alphas_cumprod[time_next]

        sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        c = (1 - alpha_next - sigma ** 2).sqrt()

        noise = torch.randn_like(output)

        output_p = x_start * alpha_next.sqrt() + \
                c * pred_noise + \
                sigma * noise
        # end "p_sample"
        x_start = ((x_start + 1) * 0.5) + EPSILON

        distri, x_aggregated = forward_imaging.forward_imaging_multinomial(x_start, pattern)
        norm = -12*distri.log_prob(pattern @ (y.reshape(y.shape[0],-1).T)/10.2).mean()
        lr = LR_MULTIPLIER * norm.item()
        
        gradients = torch.autograd.grad(norm, output)
        output = output_p - lr * gradients[0]
        output = output.detach_()
        pbar.set_postfix(norm=norm.item())

    # Post-process output
    output = ((output.detach() + 1) * 0.5)

    # Plot comparisons with unique identifier
    recon = output[0].cpu().numpy()
    recon[recon < RECON_THRESHOLD] = 0

    contorno = (ground_truth.sum(0) > 0).float().cpu().numpy()
    input_data = input_data /(input_data.max(0)[0] + EPSILON)
    plot_results(input_data*mask, recon*contorno, ground_truth)

if __name__ == "__main__":
    main()