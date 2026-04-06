
import torch
import torch.nn.functional as F

def get_velocity_gradient(u_tensor, dx=1.0, dy=1.0, dz=1.0):
    B, _, D, H, W = u_tensor.shape
    u, v, w = u_tensor[:, 0], u_tensor[:, 1], u_tensor[:, 2]
    
    def central_diff(f, axis, h):
        if axis == 'x':
            pad = F.pad(f.unsqueeze(1), (1, 1, 0, 0, 0, 0), mode='replicate').squeeze(1)
            return (pad[..., 2:] - pad[..., :-2]) / (2 * h)
        elif axis == 'y':
            pad = F.pad(f.unsqueeze(1), (0, 0, 1, 1, 0, 0), mode='replicate').squeeze(1)
            return (pad[..., 2:, :] - pad[..., :-2, :]) / (2 * h)
        elif axis == 'z':
            pad = F.pad(f.unsqueeze(1), (0, 0, 0, 0, 1, 1), mode='replicate').squeeze(1)
            return (pad[:, 2:, :, :] - pad[:, :-2, :, :]) / (2 * h)
            
    grad = torch.zeros((B, 3, 3, D, H, W), device=u_tensor.device)
    grad[:, 0, 0] = central_diff(u, 'x', dx); grad[:, 0, 1] = central_diff(u, 'y', dy); grad[:, 0, 2] = central_diff(u, 'z', dz)
    grad[:, 1, 0] = central_diff(v, 'x', dx); grad[:, 1, 1] = central_diff(v, 'y', dy); grad[:, 1, 2] = central_diff(v, 'z', dz)
    grad[:, 2, 0] = central_diff(w, 'x', dx); grad[:, 2, 1] = central_diff(w, 'y', dy); grad[:, 2, 2] = central_diff(w, 'z', dz)
    return grad

def calculate_ivd(u_tensor, dx=1.0, dy=1.0, dz=1.0):
    grad_u = get_velocity_gradient(u_tensor, dx, dy, dz)
    omega_x = grad_u[:, 2, 1] - grad_u[:, 1, 2]
    omega_y = grad_u[:, 0, 2] - grad_u[:, 2, 0]
    omega_z = grad_u[:, 1, 0] - grad_u[:, 0, 1]
    vorticity_mag = torch.sqrt(omega_x**2 + omega_y**2 + omega_z**2 + 1e-8)
    mean_vort = torch.mean(vorticity_mag, dim=(1, 2, 3), keepdim=True)
    return vorticity_mag - mean_vort

def vortex_mae_paper_loss(pred_logits, target_mask, alpha=1.0, beta=1.0, pos_weight=2.0):
    weight = torch.tensor([pos_weight], device=pred_logits.device)
    bce = F.binary_cross_entropy_with_logits(pred_logits, target_mask, pos_weight=weight)
    pred_prob = torch.sigmoid(pred_logits)
    mse = F.mse_loss(pred_prob, target_mask)
    return alpha * bce + beta * mse

def calculate_iou(pred_mask, gt_mask, threshold=0.5):
    pred = (pred_mask > threshold).float()
    intersection = (pred * gt_mask).sum()
    union = pred.sum() + gt_mask.sum() - intersection
    return (intersection + 1e-8) / (union + 1e-8)

def calculate_psnr(pred, target, max_val=1.0):
    mse = F.mse_loss(pred, target)
    if mse == 0: return torch.tensor(100.0)
    return 10 * torch.log10(max_val**2 / (mse + 1e-10))
