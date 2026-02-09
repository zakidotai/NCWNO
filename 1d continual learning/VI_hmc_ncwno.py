import hamiltorch.util as util
import numpy as np
import torch
from beartype import beartype
import torch.nn as nn
from hamiltorch import samplers
from torch.func import jacrev, functional_call
import logging
from typing import Callable, List, Union
from sens_ncwno_data import create_bayesian_forward, NCWNO1d, extract_bayesian_params
from utilities import MatReader, count_params
import os
import argparse


class VIHMCTrainer_NCWNO:
    def __init__(
            self,
            functional_model,
            mean_params,
            sensitivity_indices: Union[List, np.ndarray, torch.Tensor] = None
    ):
        """
        Prepare to train a Bayesian neural network using the hybrid VI–HMC approach.

        :param det_model: A deterministic model with the same architecture as the Bayesian model.
        :type det_model: torch.nn.Module

        :param vi_model: Bayesian model trained using variational inference.
        :type vi_model: torch.nn.Module

        :param sensitivity_function: `optional` function handle to compute sensitivity of the model. This function
        takes weights and inputs of the network and predicts output. Uses the functional call of `det_model` by default.
        :type sensitivity_function: function

        :param sensitivity_indices: `optional` list or array of indices of sensitive parameters. These indices are
        computed internally by default.
        :type sensitivity_indices: Union[List, np.ndarray, torch.Tensor]
        """

        # self.params_init = util.flatten(self.model).clone()
        self.mean_params = mean_params
        self.functional_model = functional_model
        self.sens_indices = sensitivity_indices
        # self.sensitivity_scores = None
        self.history: dict = {
            "vihmc_params": torch.inf,
            "total_params": torch.inf,
        }
        """Record of the parameter numbers. 
        - ``history["vihmc_params"]`` number of sensitive parameters sampled in the HMC step ``int``.
        - ``history["total_params"]`` total number of parameters in the model ``int``.
         """
        self.logger = logging.getLogger(__name__)

    def define_model_log_prob(
            self,
            model_loss,
            tr_data,
            prior_list,
            tau_out,
            load_prior=False,
            predict=False,
            prior_scale=1.0,
            device="cpu",
    ):
        """
        This function is built on Hamiltorch and defines the ``log_prob_func`` for ``torch.nn.Module``
        models. The resulting function is passed to the Hamiltorch sampler. This is a core component
        in workflows involving Bayesian neural networks.

        :param model_loss: Determines the likelihood model used. Options include:

                           * ``'binary_class_linear_output'`` – linear output + binary cross entropy
                           * ``'multi_class_linear_output'`` – linear output + cross entropy
                           * ``'multi_class_log_softmax_output'`` – log-softmax output + cross entropy
                           * ``'regression'`` – linear output + Gaussian likelihood (fixed variance)
                           * ``'NLL'`` – Gaussian negative log likelihood (learned variance)
                           * **function** – callable of the form ``func(y_pred, y_true)`` returning a
                             vector of shape ``(N,)``

        :type model_loss: str or function

        :param tr_data: Training dataset used to evaluate the log likelihood.
        :type tr_data: torch.utils.data.DataLoader

        :param params_flattened_list: A list containing the total number of parameters (weights/biases)
                                      per layer, in model order.
                                      Example: ``[w.nelement() for w in model.parameters()]``.
        :type params_flattened_list: list

        :param params_shape_list: A list describing the shape of each parameter tensor in the model.
                                  Example: ``[w.shape for w in model.parameters()]``.
        :type params_shape_list: list

        :param prior_list: List containing the prior precision for each layer’s parameters, assuming
                           a Gaussian prior.
        :type prior_list: list

        :param tau_out: Likelihood output precision. Relevant only when
                        ``model_loss`` is ``'regression'`` or ``'NLL'``.
                        Leave as ``1.0`` otherwise.
        :type tau_out: float

        :param load_prior: If True, load the prior distribution from a saved file.
        :type load_prior: bool

        :param predict: Set to True when invoked as part of ``hamiltorch.predict_model``.
                        Controls the number of returned objects.
        :type predict: bool

        :param prior_scale: Scaling factor applied to the prior (primarily relevant for splitting).
                            Default is ``1.0``.
        :type prior_scale: float

        :param device: Device on which computations are performed (e.g., ``'cpu'``, ``'gpu'``).
        :type device: str


        :returns: function to compute the log probabilities
        :rtype: function
        """

        dist_list = []
        if load_prior:
            dist_list.append(torch.distributions.Normal(prior_list[0], prior_list[1]))
        else:
            for tau in prior_list:
                dist_list.append(
                    torch.distributions.Normal(torch.zeros_like(tau), tau ** 0.5)
                )

        if model_loss == "NLL":
            nll_loss = torch.nn.GaussianNLLLoss(reduction="sum")

        def log_prob_func(params):

            l_prior = torch.zeros_like(
                params[0], requires_grad=True
            )  # Set l2_reg to be on the same device as params

            if load_prior:
                # for param, dist in zip(params,dist_list):
                l_prior = dist_list[0].log_prob(params).sum() + l_prior
            else:
                # Todo: add a check here
                for param, dist in zip(params, dist_list):
                    l_prior = dist.log_prob(param).sum() + l_prior

            # Code for fixing insensitive parameters at means
            weights = self.mean_params.clone()
            weights[self.sens_indices] = params

            output = []
            y = []
            for *x_batch, y_batch in tr_data:
                # Todo: add a check here
                output.append(self.functional_model(weights, (x_batch, 0)))
                y.append(y_batch)

            output = torch.cat(output)
            y = torch.cat(y)
            y_device = y.to(device)
            output = output.reshape(y_device.shape)
            assert output.shape == y_device.shape
            if model_loss == "regression":
                # crit = nn.MSELoss(reduction='mean')
                ll = -0.5 * tau_out * ((output - y_device) ** 2).sum(0)  # sum(0)
                # print(crit(output,y_device))
            elif model_loss == "binary_class_linear_output":
                crit = nn.BCEWithLogitsLoss(reduction="sum")
                ll = -tau_out * (crit(output, y_device))
            elif model_loss == "multi_class_linear_output":
                #         crit = nn.MSELoss(reduction='mean')
                crit = nn.CrossEntropyLoss(reduction="sum")
                #         crit = nn.BCEWithLogitsLoss(reduction='sum')
                ll = -tau_out * (crit(output, y_device.long().view(-1)))
                # ll = - tau_out *(torch.nn.functional.nll_loss(output, y.long().view(-1)))
            elif model_loss == "multi_class_log_softmax_output":
                ll = -tau_out * (
                    torch.nn.functional.nll_loss(output, y_device.long().view(-1))
                )

            elif model_loss == "NLL":
                ll = -nll_loss(output, y_device, tau_out * torch.ones_like(output))

            elif callable(model_loss):
                # Assume defined custom log-likelihood.
                ll = -model_loss(output, y_device).sum(0)
            else:
                raise NotImplementedError()

            if torch.cuda.is_available():
                # del x_device, y_device
                torch.cuda.empty_cache()

            if predict:
                return (ll + l_prior / prior_scale), output
            else:
                return ll + l_prior / prior_scale

        return log_prob_func

    def predict_model(
            self,
            samples,
            test_loader=None,
            model_loss="multi_class_linear_output",
            tau_out=1.0,
            prior_list=None,
    ):
        """
        This function is adapted from the Hamiltorch library and modified for DeepONets as needed.
        It produces predictions given model parameter samples. Either a dataloader **or** two tensors
        ``x`` and ``y`` may be passed—but not both.

        :param samples: A list where each element is a ``torch.Tensor`` of shape ``(D,)`` containing a
                        full parameter vector for the model. The list length ``S`` equals the number
                        of parameter samples.
        :type samples: list[torch.Tensor]

        :param test_loader: Data loader used to evaluate the samples. May be ``None`` if ``x`` and ``y``
                            are provided separately.
        :type test_loader: torch.utils.data.Dataloader or None

        :param model_loss: Determines the likelihood model used. Options include:

                           * ``'binary_class_linear_output'`` – linear output + binary cross entropy
                           * ``'multi_class_linear_output'`` – linear output + cross entropy
                           * ``'multi_class_log_softmax_output'`` – log-softmax output + cross entropy
                           * ``'regression'`` – linear output + Gaussian likelihood
                           * **function** – a callable ``func(y_pred, y_true)`` returning a vector of
                             shape ``(N,)``

        :type model_loss: str or function

        :param tau_out: Likelihood output precision. Relevant only when ``model_loss='regression'``.
                        Leave as ``1.0`` otherwise.
        :type tau_out: float

        :param prior_list: Tensor containing the prior precision for each layer’s parameters,
                           assuming a Gaussian prior.
        :type prior_list: torch.Tensor


        :returns:
            - **predictions** (``torch.Tensor``) – Model outputs of shape ``(S, N, O)``, where
              ``S`` is the number of samples, ``N`` is the number of data points, and ``O`` is the model output dimension.
            - **pred_log_prob_list** (``list``) – Log-probability values for each sample; list length is ``S``.
        :rtype: tuple
        """

        with torch.no_grad():
            params_shape_list = []
            params_flattened_list = []
            build_tau = False
            if prior_list is None:
                prior_list = []
                build_tau = True
            for weights in self.model.parameters():
                params_shape_list.append(weights.shape)
                params_flattened_list.append(weights.nelement())
                if build_tau:
                    prior_list.append(torch.tensor(1.0))

            log_prob_func = self.define_model_log_prob(
                model_loss,
                test_loader,
                prior_list,
                tau_out,
                predict=True,
                device=samples[0].device,
            )

            pred_log_prob_list = []
            pred_list = []
            for s in samples:
                lp, pred = log_prob_func(s)
                pred_log_prob_list.append(
                    lp.detach()
                )  # Side effect is to update weights to be s
                pred_list.append(pred.detach())

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return torch.stack(pred_list), pred_log_prob_list

    def run(
            self,
            train_data: torch.utils.data.DataLoader,
            valid_data: torch.utils.data.DataLoader = None,
            num_samples: int = 1000,
            num_steps: int = 30,
            step_size: float = 1e-4,
            burn: int = 0,
            loss: str = "NLL",
            tau_out: float = 1.0,
            prior_var: float = 1.0,
            load_prior: bool = False,
            init_prior: bool = False,
            sample_prior: bool = False,
            prior_file: str = None,
            device: str = "cpu",
            debug: bool = False,
    ):
        """
        Run the VI-HMC algorithm to sample from the posterior distribution of parameters.

        :param train_data: Data used to compute the log-likelihood in HMC.
        :type train_data: torch.Dataloader

        :param valid_data: Data used to validate model performance.
        :type valid_data: torch.Dataloader

        :param sens_data: `optional` Data used to compute sensitivities.
        :type sens_data: torch.Dataloader

        :param variance_threshold: Threshold defining the captured variance in the VI-HMC algorithm.
                                   This threshold determines the number of sensitive parameters.
        :type variance_threshold: float

        :param num_samples: Number of samples to draw using the VI-HMC method.
        :type num_samples: int

        :param num_steps: Number of steps to take per trajectory.
        :type num_steps: float

        :param step_size: Size of each step taken in the numerical integration.
        :type step_size: float

        :param burn: Number of samples to discard (burn-in) before collecting samples.
        :type burn: int

        :param loss: Determines the likelihood used for the model. Options include:
                     ``'binary_class_linear_output'``, ``'multi_class_linear_output'``,
                     ``'multi_class_log_softmax_output'``, ``'regression'``, ``'NLL'``,
                     or a custom function ``func(y_pred, y_true)`` returning a vector of shape (N,).
                     The options correspond to:

                     * ``'binary_class_linear_output'``: linear output + binary cross entropy.
                     * ``'multi_class_linear_output'``: linear output + cross entropy.
                     * ``'multi_class_log_softmax_output'``: log-softmax output + cross entropy.
                     * ``'regression'``: linear output + Gaussian likelihood (fixed variance).
                     * ``'NLL'``: Gaussian negative log-likelihood (learned variance).
        :type loss: str or function

        :param tau_out: Likelihood output precision. Relevant only when ``loss`` is
                        ``'regression'`` or ``'NLL'``. Interpreted as ``1/variance`` for regression
                        or as the variance for NLL. Default is ``1.0``.
        :type tau_out: float

        :param prior_var: Variance of the prior distribution.
        :type prior_var: float

        :param load_prior: If True, load the prior distribution from a saved file.
        :type load_prior: bool

        :param init_prior: If True, initialize the HMC chain using prior information.
        :type init_prior: bool

        :param sample_prior: If True, initialize HMC chains at samples drawn from the prior.
                             If False, initialize chains at the prior mean.
                             ``init_prior`` must be True for ``sample_prior`` to take effect.
        :type sample_prior: bool

        :param prior_file: Location of the prior file.
        :type prior_file: str

        :param device: Device to run the algorithm on (e.g., ``'cpu'`` or ``'gpu'``).
        :type device: str

        :param debug: If True, run HMC in debug mode.
        :type debug: bool


        :returns:
            - **params_hmc** (*list[torch.Tensor]*) – Parameter samples for the sensitive parameters.
            - **pred_list** (*list*) – Predictions for the validation data for each sample.
            - **log_prob_list** (*list*) – Log-probability values for each sample.
        :rtype: tuple
        """
        self.logger.info(
            "============================================================="
        )
        self.logger.info(
            "UQpy: Scientific Machine Learning: Performing sensitivity analysis "
        )
        self.history["vihmc_params"] = len(self.sens_indices)
        self.history["total_params"] = len(self.mean_params)
        self.logger.info(
            "UQpy: Scientific Machine Learning: Sensitivity analysis results"
        )
        self.logger.info(
            "-------------------------------------------------------------"
        )
        self.logger.info(
            f"UQpy: Scientific Machine Learning: No of total parameters: {len(self.mean_params)}"
        )
        self.logger.info(
            f"UQpy: Scientific Machine Learning: No of sensitive parameters: {len(self.sens_indices)}"
        )
        self.logger.info(
            "============================================================="
        )
        self.logger.info(
            "UQpy: Scientific Machine Learning: Performing HMC on the reduced parameters space "
        )
        self.logger.info(
            "-------------------------------------------------------------"
        )
        prior_list = []
        if load_prior:
            build_tau = False
            mean_params = torch.load(
                f"{prior_file}/means_flattened", map_location=device
            )
            std_params = torch.load(f"{prior_file}/stds_flattened", map_location=device)
            prior_list = [mean_params[self.sens_indices], std_params[self.sens_indices]]
        else:
            build_tau = True
        for _ in self.mean_params:
            if build_tau:
                prior_list.append(torch.tensor(prior_var))

        log_prob_func = self.define_model_log_prob(
            loss,
            train_data,
            prior_list,
            tau_out,
            device=device,
        )

        if init_prior:
            learned_mus = torch.load(
                f"{prior_file}/means_flattened", map_location=device
            )
            learned_sigmas = torch.load(
                f"{prior_file}/stds_flattened", map_location=device
            )
            params_trained = (
                torch.normal(learned_mus, learned_sigmas)
                if sample_prior
                else learned_mus
            )
        else:
            # Todo: Initialize at random parameter values
            params_trained = self.mean_params

        params_init = params_trained[self.sens_indices].clone()
        params_hmc = samplers.sample(
            log_prob_func,
            params_init,
            num_samples=num_samples,
            num_steps_per_sample=num_steps,
            step_size=step_size,
            debug=debug,
            sampler=samplers.Sampler.HMC,
        )
        if valid_data is not None:
            pred_list, log_prob_list = self.predict_model(
                params_hmc[burn:],
                valid_data,
                model_loss=loss,
                tau_out=tau_out,
                prior_list=prior_list,
            )
            self.logger.info("UQpy: Scientific Machine Learning: Completed VI-HMC ")
            return params_hmc, pred_list, log_prob_list
        self.logger.info("UQpy: Scientific Machine Learning: Completed VI-HMC ")
        return params_hmc


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
    forward_fn = create_bayesian_forward(model, device)
    sensitivity_indices = [0]
    vihmc_trainer = VIHMCTrainer_NCWNO(forward_fn, mu_params, sensitivity_indices)
    vihmc_trainer.run(train_loaders)


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
