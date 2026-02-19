"""
Sensitivity Analysis for NCWNO1d Bayesian Model

This script performs sensitivity analysis on the Bayesian NCWNO1d model
trained for 1D continual learning of PDEs.

It computes the sensitivity of model outputs to each Bayesian parameter,
helping identify which parameters are most important for the model's predictions.

Usage:
    python sens_ncwno.py [--model_path PATH] [--num_batches NUM] [--var_threshold THRESH]
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Note: jacrev from torch.func is not used because it's incompatible with
# pytorch_wavelets (which uses autograd.Function without setup_context)

# Add parent directory for imports
directory = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.insert(0, os.getcwd())

from utilities import MatReader, LpLoss, count_params
from ncwno_modules import WaveConv1d, WaveEncoder1d, Gate_context1d

# Import UQpy for Bayesian layers
import UQpy.scientific_machine_learning as sml
from UQpy.scientific_machine_learning.baseclass import NormalBayesianLayer

torch.manual_seed(0)
np.random.seed(0)

# %%
""" Model Definition (same as training script) """


class Expert_WNO(nn.Module):
    def __init__(self, level, width, expert_num, size):
        super(Expert_WNO, self).__init__()
        self.level = level
        self.width = width
        self.expert_num = expert_num

        wavelet = ['db' + str(i + 1) for i in range(self.expert_num)]
        self.Expert_layers0 = WaveConv1d(self.width, self.width, self.level, size, wavelet[0])
        self.Expert_layers1 = WaveConv1d(self.width, self.width, self.level, size, wavelet[1])
        self.Expert_layers2 = WaveConv1d(self.width, self.width, self.level, size, wavelet[2])
        self.Expert_layers3 = WaveConv1d(self.width, self.width, self.level, size, wavelet[3])
        self.Expert_layers4 = WaveConv1d(self.width, self.width, self.level, size, wavelet[4])
        self.Expert_layers5 = WaveConv1d(self.width, self.width, self.level, size, wavelet[5])
        self.Expert_layers6 = WaveConv1d(self.width, self.width, self.level, size, wavelet[6])
        self.Expert_layers7 = WaveConv1d(self.width, self.width, self.level, size, wavelet[7])
        self.Expert_layers8 = WaveConv1d(self.width, self.width, self.level, size, wavelet[8])
        self.Expert_layers9 = WaveConv1d(self.width, self.width, self.level, size, wavelet[9])

    def forward(self, x, lambda_):
        x = lambda_[..., 0:1] * self.Expert_layers0(x) + lambda_[..., 1:2] * self.Expert_layers1(x) + \
            lambda_[..., 2:3] * self.Expert_layers2(x) + lambda_[..., 3:4] * self.Expert_layers3(x) + \
            lambda_[..., 4:5] * self.Expert_layers4(x) + lambda_[..., 5:6] * self.Expert_layers5(x) + \
            lambda_[..., 6:7] * self.Expert_layers6(x) + lambda_[..., 7:8] * self.Expert_layers7(x) + \
            lambda_[..., 8:9] * self.Expert_layers8(x) + lambda_[..., 9:10] * self.Expert_layers9(x)
        return x


class NCWNO1d(nn.Module):
    def __init__(self, width, level, input_dim, hidden_dim, space_len, expert_num, label_lifting, size, padding=0,
                 is_bayesian=False):
        super(NCWNO1d, self).__init__()
        self.level = level
        self.width = width
        self.hidden_dim = hidden_dim
        self.space_len = space_len
        self.padding = padding
        self.size = size
        self.expert_num = expert_num
        self.label_lifting = label_lifting
        self.conv_layers = nn.ModuleList()
        self.w_layers = nn.ModuleList()
        self.gate = nn.ModuleList()

        for hdim in range(self.hidden_dim):
            self.gate.append(Gate_context1d(width, width, expert_num, label_lifting, size, is_bayesian=is_bayesian))

        self.fc0 = nn.Conv1d(input_dim, self.width, 1)
        self.fc1 = nn.Conv1d(self.width, self.width, 1)
        for hdim in range(self.hidden_dim):
            self.conv_layers.append(Expert_WNO(self.level, self.width, self.expert_num, self.size))
            self.w_layers.append(nn.Conv1d(self.width, self.width, 1))

        self.fc2 = nn.Conv1d(self.width, 128, 1)
        self.fc3 = nn.Conv1d(128, 1, 1)

    def forward(self, x, label):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=1)
        x = self.fc0(x)
        x = self.fc1(x)
        if self.padding != 0:
            x = F.pad(x, [0, self.padding])

        lambda_ = []
        label = self.get_label(label, x.shape, x.device)
        for gate_ in self.gate:
            lambda_.append(gate_(x, label))

        for wib, w0, lam in zip(self.conv_layers, self.w_layers, lambda_):
            x = wib(x, lam) + w0(x)
            x = F.mish(x)

        if self.padding != 0:
            x = x[..., :-self.padding]
        x = self.fc2(x)
        x = F.mish(x)
        x = self.fc3(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[-1]
        gridx = torch.tensor(np.linspace(0, self.space_len, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x).repeat([batchsize, 1, 1])
        return gridx.to(device)

    def get_label(self, label, shape, device):
        batchsize, channel_size, size_x = shape
        label = label.repeat(batchsize, channel_size, 1).to(device)
        return label.float()


def get_bayesian_layer_info(model):
    """
    Extract information about all Bayesian layers in the model.
    
    Returns:
        bayesian_layers: List of (name, module) tuples for Bayesian layers
        param_info: List of dicts with parameter shapes and names
    """
    bayesian_layers = []
    param_info = []

    for name, module in model.named_modules():
        if isinstance(module, sml.BayesianLinear):
            bayesian_layers.append((name, module))
            info = {
                'name': name,
                'in_features': module.in_features,
                'out_features': module.out_features,
                'weight_shape': (module.out_features, module.in_features),
                'bias_shape': (module.out_features,) if module.bias_mu is not None else None
            }
            param_info.append(info)

    return bayesian_layers, param_info


def extract_bayesian_params(model):
    """
    Extract mean (mu) and standard deviation (sigma) parameters from all Bayesian layers.
    
    Returns:
        mu_params: Flattened tensor of all mean parameters
        sigma_params: Flattened tensor of all sigma parameters (computed from rho)
        param_shapes: List of shapes for reconstruction
    """
    mu_list = []
    sigma_list = []
    param_shapes = []

    for name, module in model.named_modules():
        if isinstance(module, sml.BayesianLinear):
            # Weight parameters
            mu_list.append(module.weight_mu.flatten())
            # sigma = log(1 + exp(rho)) - softplus transformation
            sigma_list.append(torch.log1p(torch.exp(module.weight_rho)).flatten())
            param_shapes.append(('weight', name, module.weight_mu.shape))

            # Bias parameters (if present)
            if module.bias_mu is not None:
                mu_list.append(module.bias_mu.flatten())
                sigma_list.append(torch.log1p(torch.exp(module.bias_rho)).flatten())
                param_shapes.append(('bias', name, module.bias_mu.shape))

    mu_params = torch.cat(mu_list)
    sigma_params = torch.cat(sigma_list)

    return mu_params, sigma_params, param_shapes


def create_bayesian_forward(model, T0, step, T):
    """
    Create a forward function that takes explicit Bayesian parameters.
    
    This function creates a closure that uses the provided mean parameters
    to perform forward pass through the Bayesian layers.
    """

    # Get the structure of Bayesian layers
    bayesian_layer_names = []
    bayesian_layer_configs = []  # Store (name, weight_shape, bias_shape)

    for name, module in model.named_modules():
        if isinstance(module, sml.BayesianLinear):
            bayesian_layer_names.append(name)
            weight_shape = (module.out_features, module.in_features)
            bias_shape = (module.out_features,) if module.bias_mu is not None else None
            bayesian_layer_configs.append((name, weight_shape, bias_shape))

    def forward_with_params(mu_params, model_input):
        """
        Forward pass using explicit mean parameters for Bayesian layers.
        
        Args:
            mu_params: Flattened tensor of all Bayesian mean parameters
            model_input: Tuple of (x, label) for the model
        
        Returns:
            Model output (averaged over spatial dimension for sensitivity computation)
        """
        x, label = model_input

        def model_forward(x_forward, label_forward):
            # Reconstruct parameters into a dict
            param_dict = {}
            idx = 0
            for name, weight_shape, bias_shape in bayesian_layer_configs:
                w_size = weight_shape[0] * weight_shape[1]
                param_dict[f'{name}.weight'] = mu_params[idx:idx + w_size].view(weight_shape)
                idx += w_size
                if bias_shape is not None:
                    b_size = bias_shape[0]
                    param_dict[f'{name}.bias'] = mu_params[idx:idx + b_size].view(bias_shape)
                    idx += b_size

            # Forward through the model with explicit parameters
            grid = model.get_grid(x_forward.shape, x_forward.device)
            x_in = torch.cat((x_forward, grid), dim=1)
            x_out = model.fc0(x_in)
            x_out = model.fc1(x_out)

            if model.padding != 0:
                x_out = F.pad(x_out, [0, model.padding])

            # Compute gate outputs with explicit parameters
            label_expanded = model.get_label(label_forward, x_out.shape, x_out.device)
            lambda_list = []

            for gate_idx, gate_module in enumerate(model.gate):
                # Gate forward with explicit Bayesian parameters
                lambda_0 = gate_module.lifting_network(label_expanded)
                lambda_1 = gate_module.wno_encode(x_out)
                gate_input = torch.cat((lambda_0, lambda_1), dim=-1)

                # Process through gate Sequential with explicit params
                h = gate_input
                bayesian_layer_idx = 0
                for layer_idx, layer in enumerate(gate_module.gate):
                    if isinstance(layer, sml.BayesianLinear):
                        # Get the layer name in the model
                        layer_name = f'gate.{gate_idx}.gate.{layer_idx}'
                        weight = param_dict[f'{layer_name}.weight']
                        bias = param_dict.get(f'{layer_name}.bias', None)
                        h = F.linear(h, weight, bias)
                        bayesian_layer_idx += 1
                    else:
                        # Non-Bayesian layer (activation, softmax)
                        h = layer(h)

                lambda_list.append(h)

            # Process through conv layers
            for wib, w0, lam in zip(model.conv_layers, model.w_layers, lambda_list):
                x_out = wib(x_out, lam) + w0(x_out)
                x_out = F.mish(x_out)

            if model.padding != 0:
                x_out = x_out[..., :-model.padding]

            x_out = model.fc2(x_out)
            x_out = F.mish(x_out)
            x_out = model.fc3(x_out)

            # Return mean over spatial dimension for sensitivity computation
            return x_out

        for t in range(T0, T0+T, step):
            im = model_forward(x, label)
            if t == T0:
                pred = im
            else:
                pred = torch.cat((pred, im), 1)
            x = torch.cat((x[:, step:, ...], im), dim=1)

        return pred

    return forward_with_params


def eval_jacobian(forward_fn, x, label, mu_params):
    """
    Compute the Jacobian of the forward function w.r.t. Bayesian mean parameters.
    
    NOTE: This function uses functorch's jacrev which is incompatible with
    pytorch_wavelets (which uses autograd.Function without setup_context).
    Use the gradient-based method instead (use_jacobian=False).
    
    Args:
        forward_fn: Forward function that takes (mu_params, (x, label))
        x: Input tensor
        label: Label tensor
        mu_params: Mean parameters tensor
    
    Returns:
        Sensitivity scores (mean squared gradients)
    """
    raise RuntimeError(
        "jacrev (functorch) is incompatible with pytorch_wavelets. "
        "The wavelet transforms use autograd.Function without setup_context, "
        "which is required for functorch transforms. "
        "Please use --use_gradient flag to use the gradient-based method instead."
    )


def eval_gradients_per_output(forward_fn, x, label, mu_params_grad, pbar_outputs=None):
    """
    Compute squared gradients using per-output-element backward passes.
    
    This is memory efficient and avoids the cancellation problem that occurs
    when summing outputs before backward (positive/negative gradients cancel).
    
    Instead of computing the full Jacobian, we:
    1. Forward pass to get output
    2. For each output element, compute gradients separately
    3. Square and accumulate gradients
    
    Args:
        forward_fn: Forward function that takes (mu_params, (x, label))
        x: Model input tensor
        label: Label tensor
        mu_params_grad: Mean parameters tensor with requires_grad=True
        pbar_outputs: Optional tqdm progress bar for output elements
    
    Returns:
        Mean squared gradients: (num_params,)
    """
    # Forward pass
    output = forward_fn(mu_params_grad, (x, label))

    # Flatten to iterate over each output element
    output_flat = output.flatten()
    num_outputs = output_flat.numel()

    # Accumulate squared gradients
    grads_squared_sum = torch.zeros_like(mu_params_grad)

    # Update progress bar total if provided
    if pbar_outputs is not None:
        pbar_outputs.reset(total=num_outputs)

    for i in range(num_outputs):
        # Zero gradients
        if mu_params_grad.grad is not None:
            mu_params_grad.grad.zero_()

        # Backward for this output element
        # retain_graph=True because we need to backprop through the same graph multiple times
        output_flat[i].backward(retain_graph=(i < num_outputs - 1))

        # Accumulate squared gradients
        grads_squared_sum += mu_params_grad.grad ** 2

        # Update progress bar
        if pbar_outputs is not None:
            pbar_outputs.update(1)

    # Mean over output elements (same as in original jacrev approach)
    grads_squared_mean = grads_squared_sum / num_outputs

    # Zero gradients for next batch
    if mu_params_grad.grad is not None:
        mu_params_grad.grad.zero_()

    return grads_squared_mean


def compute_sensitivity(data_loader, forward_fn, mu_params, sigma_params,
                        label, device, max_batches=None, use_jacobian=False,
                        use_per_output=True):
    """
    Compute sensitivity scores over the entire dataset.
    
    Args:
        data_loader: DataLoader for training data
        forward_fn: Forward function
        mu_params: Mean parameters
        sigma_params: Sigma parameters
        label: PDE label
        device: Computation device
        max_batches: Maximum number of batches to process (None for all)
        use_jacobian: If True, use full Jacobian computation (NOT SUPPORTED - raises error).
                     If False, use gradient-based method.
        use_per_output: If True (default), compute gradients per output element to avoid
                       cancellation of positive/negative gradients. This is more accurate
                       but slower. If False, sum outputs before backward (can have
                       cancellation issues).
    
    Returns:
        sensitivity_scores: Sensitivity of each parameter
    """
    grads_sum = torch.zeros_like(mu_params)
    num_batches = len(data_loader) if max_batches is None else min(max_batches, len(data_loader))

    print(f'Computing sensitivity over {num_batches} batches...')

    if use_jacobian:
        # This will raise an error explaining why jacrev doesn't work
        eval_jacobian(forward_fn, None, None, None)

    if use_per_output:
        print(f'Using per-output gradient method (avoids gradient cancellation)')
    else:
        print(f'Using summed gradient method (WARNING: may have gradient cancellation)')

    # Gradient-based sensitivity computation
    # This uses regular PyTorch autograd which works with pytorch_wavelets
    mu_params_grad = mu_params.clone().detach().requires_grad_(True)

    # Create progress bars with elapsed time and ETA
    # Outer progress bar for batches
    pbar_batches = tqdm(data_loader, total=num_batches, desc='Batches',
                        unit='batch', position=0, leave=True,
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

    # Inner progress bar for output elements (only used with per_output method)
    pbar_outputs = None
    if use_per_output:
        pbar_outputs = tqdm(total=0, desc='  Outputs', unit='elem', position=1, leave=False,
                            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

    for batch_idx, (xx, yy) in enumerate(pbar_batches):
        if max_batches is not None and batch_idx >= max_batches:
            break

        xx = xx.to(device)

        if use_per_output:
            # Per-output gradient computation (avoids cancellation)
            grads_squared = eval_gradients_per_output(forward_fn, xx, label, mu_params_grad, pbar_outputs)
            grads_sum += grads_squared / num_batches
        else:
            # Original method: sum outputs before backward (can have cancellation)
            # Forward pass
            output = forward_fn(mu_params_grad, (xx, label))
            # Sum over all outputs to get a scalar loss
            loss = output.sum()

            # Backward pass to compute gradients
            loss.backward()

            # Accumulate squared gradients
            grads_sum += (mu_params_grad.grad ** 2) / num_batches

            # Zero gradients for next iteration
            mu_params_grad.grad.zero_()

    # Close progress bars
    pbar_batches.close()
    if pbar_outputs is not None:
        pbar_outputs.close()

    # Sensitivity = (gradient^2) * (variance) = grad^2 * sigma^2
    sensitivity_scores = grads_sum * (sigma_params ** 2)

    return sensitivity_scores


def analyze_sensitivity(sensitivity_scores, param_shapes, var_threshold=0.9):
    """
    Analyze sensitivity scores to identify important parameters.
    
    Args:
        sensitivity_scores: Tensor of sensitivity scores
        param_shapes: List of parameter shape info
        var_threshold: Threshold for cumulative variance explained
    
    Returns:
        num_sensitive: Number of parameters explaining var_threshold variance
        sensitive_indices: Indices of sensitive parameters
        analysis_results: Dict with detailed analysis
    """
    total_var = torch.sum(sensitivity_scores)
    sorted_sens, sorted_indices = torch.sort(sensitivity_scores, descending=True)
    cumulative_var = torch.cumsum(sorted_sens, 0)

    # Find number of parameters explaining var_threshold variance
    num_sensitive = torch.sum(cumulative_var / total_var <= var_threshold).item() + 1
    sensitive_indices = sorted_indices[:num_sensitive].sort()[0]

    # Analyze by layer
    layer_sensitivity = {}
    idx = 0
    for param_type, layer_name, shape in param_shapes:
        size = np.prod(shape)
        layer_key = f'{layer_name}.{param_type}'
        layer_sensitivity[layer_key] = {
            'total_sensitivity': sensitivity_scores[idx:idx + size].sum().item(),
            'mean_sensitivity': sensitivity_scores[idx:idx + size].mean().item(),
            'max_sensitivity': sensitivity_scores[idx:idx + size].max().item(),
            'num_params': size
        }
        idx += size

    # Sort layers by total sensitivity
    sorted_layers = sorted(layer_sensitivity.items(),
                           key=lambda x: x[1]['total_sensitivity'],
                           reverse=True)

    return num_sensitive, sensitive_indices, {
        'total_variance': total_var.item(),
        'num_sensitive_params': num_sensitive,
        'total_params': len(sensitivity_scores),
        'sensitive_fraction': num_sensitive / len(sensitivity_scores),
        'layer_sensitivity': dict(sorted_layers)
    }


def verify_forward_function(model, forward_fn, mu_params, x_sample, label, T0, step, T, device):
    """
    Verify that the custom forward function produces the same output as the model.
    """
    model.eval()
    for module in model.modules():
        if hasattr(module, "sampling"):
            module.Sampling = False
    with torch.no_grad():
        # Original model output
        xx = x_sample.clone()
        for t in range(T0, T+T0, step):
            im = model(xx, label)
            if t == T0:
                pred = im
            else:
                pred = torch.cat((pred, im), 1)
            xx = torch.cat((xx[:, step:, ...], im), dim=1)
        original_output = pred

        # Custom forward function output
        custom_output = forward_fn(mu_params, (x_sample, label))

        # Check if outputs are close (note: Bayesian layers sample, so there may be differences)
        diff = torch.abs(original_output - custom_output).mean().item()

    print(f'  Forward function verification:')
    print(f'    Original output shape: {original_output.shape}')
    print(f'    Custom output shape: {custom_output.shape}')
    print(f'    Mean absolute difference: {diff:.6f}')

    if diff > 1.0:
        print('  WARNING: Large difference detected. The custom forward function may have issues.')
    else:
        print('  Forward function verification PASSED.')

    return diff


def main(args):
    """Main function for sensitivity analysis."""

    # Device setup
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Model configuration (same as training)
    T = 20
    T0 = 10
    step = 1
    sub = 2
    S = 256
    level = 4
    width = 128
    batch_size = 20
    ntrain = 1000

    # Data paths
    data_paths = [
        'data/Allen_Cahn_1D_pde_x512_T50_N1500_v1em4.mat',
        'data/Nagumo_1D_pde_x512_T50_N1500.mat',
        'data/Wave_1D_pde_x512_T50_N1500_c2.mat',
    ]
    case_len = len(data_paths)
    data_label = torch.arange(1, case_len + 1)

    # Load data
    print('Loading data...')
    data = []
    for path in data_paths:
        if os.path.exists(path):
            print(f'  Loading: {path}')
            data.append((MatReader(path).read_field('sol')[::sub, :, :]).permute(2, 1, 0))
        else:
            print(f'  WARNING: Data file not found: {path}')

    if len(data) == 0:
        raise FileNotFoundError("No data files found. Please check data paths.")

    # Create data loaders
    train_a = [d[:ntrain, :T0, :] for d in data]
    train_u = [d[:ntrain, T0:T0 + T, :] for d in data]

    train_loaders = [
        torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_a[i], train_u[i]),
            batch_size=batch_size, shuffle=False
        ) for i in range(len(data))
    ]

    # Load model
    print(f'\nLoading model from: {args.model_path}')
    # weights_only=False is needed for loading full model objects saved with torch.save(model, path)
    # This is safe when loading your own trained models
    loaded_data = torch.load(args.model_path, map_location=device, weights_only=False)

    if isinstance(loaded_data, nn.Module):
        model = loaded_data
        print('  Loaded full model object.')
    elif isinstance(loaded_data, dict):
        if 'model_state_dict' in loaded_data:
            state_dict = loaded_data['model_state_dict']
        else:
            state_dict = loaded_data

        # Check if Bayesian model
        is_bayesian = any('weight_mu' in key or 'weight_rho' in key
                          for key in state_dict.keys())

        if is_bayesian:
            print('  Detected Bayesian model.')
            model = NCWNO1d(width=width, level=level, input_dim=T0 + 1, hidden_dim=4,
                            space_len=1, expert_num=10, label_lifting=2 ** 4, size=S,
                            is_bayesian=True).to(device)
        else:
            raise ValueError("Model does not appear to be Bayesian. Cannot perform sensitivity analysis.")

        model.load_state_dict(state_dict, strict=False)
        print('  Model loaded from state_dict.')
    else:
        raise ValueError(f'Unexpected data type: {type(loaded_data)}')

    model.eval()
    print(f'Total parameters: {count_params(model):,}')

    # Extract Bayesian parameters
    print('\nExtracting Bayesian parameters...')
    mu_params, sigma_params, param_shapes = extract_bayesian_params(model)
    print(f'  Total Bayesian parameters: {len(mu_params):,}')
    print(f'  Parameter groups: {len(param_shapes)}')

    # Create forward function
    print('\nCreating forward function for Jacobian computation...')
    forward_fn = create_bayesian_forward(model, T0, step, T)

    # Verify forward function with a sample batch
    print('\nVerifying forward function...')
    sample_batch = next(iter(train_loaders[0]))
    sample_x = sample_batch[0][:2].to(device)  # Take 2 samples
    verify_forward_function(model, forward_fn, mu_params, sample_x, data_label[0], T0, step, T, device)

    # Compute sensitivity for each PDE case
    all_sensitivity_scores = []
    all_analysis_results = []

    for case_idx in range(len(data)):
        print(f'\n{"=" * 60}')
        print(f'Computing sensitivity for PDE case {case_idx} ({data_paths[case_idx].split("/")[-1]})')
        print(f'{"=" * 60}')

        label = data_label[case_idx]

        sensitivity_scores = compute_sensitivity(
            train_loaders[case_idx],
            forward_fn,
            mu_params,
            sigma_params,
            label,
            device,
            max_batches=args.num_batches,
            use_jacobian=False,  # jacrev is incompatible with pytorch_wavelets
            use_per_output=not args.use_summed_gradient  # Per-output avoids cancellation
        )

        num_sensitive, sensitive_indices, analysis = analyze_sensitivity(
            sensitivity_scores, param_shapes, args.var_threshold
        )

        all_sensitivity_scores.append(sensitivity_scores)
        all_analysis_results.append(analysis)

        print(f'\nResults for PDE case {case_idx}:')
        print(f'  Total variance: {analysis["total_variance"]:.6f}')
        print(f'  Sensitive parameters: {num_sensitive:,} / {len(mu_params):,} '
              f'({100 * analysis["sensitive_fraction"]:.2f}%)')
        print(f'\n  Top 5 most sensitive layers:')
        for layer_name, stats in list(analysis['layer_sensitivity'].items())[:5]:
            print(f'    {layer_name}: total={stats["total_sensitivity"]:.6f}, '
                  f'mean={stats["mean_sensitivity"]:.6f}, params={stats["num_params"]}')

    # Save results
    output_dir = 'results_sensitivity'
    os.makedirs(output_dir, exist_ok=True)

    # Combine sensitivity across all PDEs
    combined_sensitivity = torch.stack(all_sensitivity_scores).mean(dim=0)
    num_sensitive, sensitive_indices, combined_analysis = analyze_sensitivity(
        combined_sensitivity, param_shapes, args.var_threshold
    )

    print(f'\n{"=" * 60}')
    print('Combined sensitivity analysis (averaged over all PDEs):')
    print(f'{"=" * 60}')
    print(f'  Sensitive parameters: {num_sensitive:,} / {len(mu_params):,} '
          f'({100 * combined_analysis["sensitive_fraction"]:.2f}%)')

    # Save sensitivity scores
    save_path = os.path.join(output_dir, 'sensitivity_scores_ncwno.pt')
    torch.save({
        'combined_sensitivity': combined_sensitivity,
        'per_pde_sensitivity': all_sensitivity_scores,
        'sensitive_indices': sensitive_indices,
        'num_sensitive': num_sensitive,
        'param_shapes': param_shapes,
        'analysis_results': all_analysis_results,
        'combined_analysis': combined_analysis,
        'var_threshold': args.var_threshold,
        'model_path': args.model_path
    }, save_path)
    print(f'\nSensitivity scores saved to: {save_path}')

    # Print summary
    print(f'\n{"=" * 60}')
    print('SUMMARY')
    print(f'{"=" * 60}')
    print(f'Model: {args.model_path}')
    print(f'Total Bayesian parameters: {len(mu_params):,}')
    print(f'Sensitive parameters ({100 * args.var_threshold:.0f}% variance): {num_sensitive:,}')
    print(f'Reduction: {100 * (1 - num_sensitive / len(mu_params)):.1f}%')

    return combined_sensitivity, sensitive_indices, combined_analysis


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sensitivity Analysis for NCWNO1d Bayesian Model')
    parser.add_argument('--model_path', type=str,
                        default='data/model/Foundation_1d_10exp_0_bayesian_bs_20',
                        help='Path to the trained Bayesian model')
    parser.add_argument('--num_batches', type=int, default=None,
                        help='Number of batches to use (None for all)')
    parser.add_argument('--var_threshold', type=float, default=0.9,
                        help='Variance threshold for sensitive parameter selection')
    parser.add_argument('--use_gradient', action='store_true',
                        help='[DEPRECATED] Gradient-based method is now the default. '
                             'jacrev is incompatible with pytorch_wavelets.')
    parser.add_argument('--use_summed_gradient', action='store_true',
                        help='Use summed gradient method instead of per-output method. '
                             'WARNING: summed method can have gradient cancellation issues. '
                             'Per-output method (default) is more accurate but slower.')

    args = parser.parse_args()

    sensitivity_scores, sensitive_indices, analysis = main(args)