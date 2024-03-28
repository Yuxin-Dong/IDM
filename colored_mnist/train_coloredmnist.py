# This script is developed based on https://github.com/alexrame/fishr/blob/main/coloredmnist/train_coloredmnist.py
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import random
import argparse
import numpy as np
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torchvision import datasets
from torch import nn, optim, autograd

from backpack import backpack, extend
from backpack.extensions import BatchGrad

parser = argparse.ArgumentParser(description='Colored MNIST')

# select your algorithm
parser.add_argument(
    '--algorithm',
    type=str,
    default="fishr",
    choices=[
        'erm',    # Empirical Risk Minimization
        'irm',    # Invariant Risk Minimization (https://arxiv.org/abs/1907.02893)
        'rex',    # Out-of-Distribution Generalization via Risk Extrapolation (https://icml.cc/virtual/2021/oral/9186)
        'iga',    # When is invariance useful in an Out-of-Distribution Generalization problem? (https://arxiv.org/abs/2008.01883)
        'fishr',  # Fishr: Invariant Gradient Variances for Out-of-Distribution Generalization (https://icml.cc/virtual/2022/poster/17213)
        'idm',    # Provable Domain Generalization via Information Theory Guided Distribution Matching (Ours)
    ]
)

parser.add_argument('--label_flipping_prob', type=float, default=0.25)
parser.add_argument('--hidden_dim', type=int, default=433)
parser.add_argument('--l2_regularizer_weight', type=float, default=0.00034)
parser.add_argument('--lr', type=float, default=0.000449)
parser.add_argument('--penalty_anneal_iters', type=int, default=154)
parser.add_argument('--penalty_weight', type=float, default=2888595.180638)
parser.add_argument('--steps', type=int, default=501)
parser.add_argument('--margin', type=int, default=1)
parser.add_argument('--grayscale_model', action='store_true')
parser.add_argument('--n_restarts', type=int, default=10)
parser.add_argument('--seed', type=int, default=0, help='Seed for everything')

flags = parser.parse_args()

print('Flags:')
for k, v in sorted(vars(flags).items()):
    print("\t{}: {}".format(k, v))

random.seed(flags.seed)
np.random.seed(flags.seed)
torch.manual_seed(flags.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

final_train_accs = []
final_test_accs = []
final_graytest_accs = []
for restart in range(flags.n_restarts):
    print("Restart", restart)

    # Load MNIST, make train/val splits, and shuffle train set examples

    mnist = datasets.MNIST('~/datasets/mnist', train=True, download=True)
    mnist_train = (mnist.data[:50000], mnist.targets[:50000])
    mnist_val = (mnist.data[50000:], mnist.targets[50000:])

    rng_state = np.random.get_state()
    np.random.shuffle(mnist_train[0].numpy())
    np.random.set_state(rng_state)
    np.random.shuffle(mnist_train[1].numpy())

    # Build environments

    def make_environment(images, labels, e, grayscale=False):

        def torch_bernoulli(p, size):
            return (torch.rand(size) < p).float()

        def torch_xor(a, b):
            return (a - b).abs()  # Assumes both inputs are either 0 or 1

        # 2x subsample for computational convenience
        images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit; flip label with probability 0.25
        labels = (labels < 5).float()
        labels = torch_xor(labels, torch_bernoulli(flags.label_flipping_prob, len(labels)))
        # Assign a color based on the label; flip the color with probability e
        colors = torch_xor(labels, torch_bernoulli(e, len(labels)))
        # Apply the color to the image by zeroing out the other color channel
        images = torch.stack([images, images], dim=1)
        if not grayscale:
            images[torch.tensor(range(len(images))), (1 - colors).long(), :, :] *= 0
        return {'images': (images.float() / 255.).cuda(), 'labels': labels[:, None].cuda()}

    envs = [
        make_environment(mnist_train[0][::2], mnist_train[1][::2], 0.2),
        make_environment(mnist_train[0][1::2], mnist_train[1][1::2], 0.1),
        make_environment(mnist_val[0], mnist_val[1], 0.9),
        make_environment(mnist_val[0], mnist_val[1], 0.9, grayscale=True)
    ]

    # Define and instantiate the model

    class MLP(nn.Module):

        def __init__(self):
            super(MLP, self).__init__()
            if flags.grayscale_model:
                lin1 = nn.Linear(14 * 14, flags.hidden_dim)
            else:
                lin1 = nn.Linear(2 * 14 * 14, flags.hidden_dim)
            lin2 = nn.Linear(flags.hidden_dim, flags.hidden_dim)

            self.classifier = extend(nn.Linear(flags.hidden_dim, 1))
            for lin in [lin1, lin2, self.classifier]:
                nn.init.xavier_uniform_(lin.weight)
                nn.init.zeros_(lin.bias)
            self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True))

        def forward(self, input):
            if flags.grayscale_model:
                out = input.view(input.shape[0], 2, 14 * 14).sum(dim=1)
            else:
                out = input.view(input.shape[0], 2 * 14 * 14)
            features = self._main(out)
            logits = self.classifier(features)
            return features, logits

    mlp = MLP().cuda()

    # Define loss function helpers

    def mean_nll(logits, y):
        return nn.functional.binary_cross_entropy_with_logits(logits, y)

    def mean_accuracy(logits, y):
        preds = (logits > 0.).float()
        return ((preds - y).abs() < 1e-2).float().mean()

    def compute_irm_penalty(logits, y):
        scale = torch.tensor(1.).cuda().requires_grad_()
        loss = mean_nll(logits * scale, y)
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad**2)

    bce_extended = extend(nn.BCEWithLogitsLoss())

    def compute_sorted_grads(features, labels, classifier):
        logits = classifier(features)
        loss = bce_extended(logits, labels)
        with backpack(BatchGrad()):
            loss.backward(
                inputs=list(classifier.parameters()), retain_graph=True, create_graph=True
            )

        dict_grads = OrderedDict(
            [
                (name, weights.grad_batch.clone().view(weights.grad_batch.size(0), -1))
                for name, weights in classifier.named_parameters()
            ]
        )
        for name, _grads in dict_grads.items():
            grads = _grads * labels.size(0)  # multiply by batch size
            dict_grads[name] = grads.sort(dim=0).values[flags.margin:-flags.margin, :]

        return dict_grads

    def compute_grads_variance(features, labels, classifier):
        logits = classifier(features)
        loss = bce_extended(logits, labels)
        with backpack(BatchGrad()):
            loss.backward(
                inputs=list(classifier.parameters()), retain_graph=True, create_graph=True
            )

        dict_grads = OrderedDict(
            [
                (name, weights.grad_batch.clone().view(weights.grad_batch.size(0), -1))
                for name, weights in classifier.named_parameters()
            ]
        )
        dict_grads_variance = {}
        for name, _grads in dict_grads.items():
            grads = _grads * labels.size(0)  # multiply by batch size
            env_mean = grads.mean(dim=0, keepdim=True)
            if flags.algorithm != "fishr_notcentered":
                grads = grads - env_mean
            if flags.algorithm == "fishr_offdiagonal":
                dict_grads_variance[name] = torch.einsum("na,nb->ab", grads,
                                                    grads) / (grads.size(0) * grads.size(1))
            else:
                dict_grads_variance[name] = (grads).pow(2).mean(dim=0)

        return dict_grads_variance

    def l2_between_grads_expectation(grads_1, grads_2):
        assert len(grads_1) == len(grads_2)
        grads_1_values = [grads_1[key].mean(0) for key in sorted(grads_1.keys())]
        grads_2_values = [grads_2[key].mean(0) for key in sorted(grads_2.keys())]
        return (
            torch.cat(tuple([t.view(-1) for t in grads_1_values])) -
            torch.cat(tuple([t.view(-1) for t in grads_2_values]))
        ).pow(2).mean()

    def l2_between_grads_variance(cov_1, cov_2):
        assert len(cov_1) == len(cov_2)
        cov_1_values = [cov_1[key] for key in sorted(cov_1.keys())]
        cov_2_values = [cov_2[key] for key in sorted(cov_2.keys())]
        return (
            torch.cat(tuple([t.view(-1) for t in cov_1_values])) -
            torch.cat(tuple([t.view(-1) for t in cov_2_values]))
        ).pow(2).sum()

    def per_sample_distribution_matching(grads_1, grads_2):
        assert len(grads_1) == len(grads_2)
        grads_1_values = [grads_1[key] for key in sorted(grads_1.keys())]
        grads_2_values = [grads_2[key] for key in sorted(grads_2.keys())]
        return (
            torch.cat(tuple([t.view(-1) for t in grads_1_values])) -
            torch.cat(tuple([t.view(-1) for t in grads_2_values]))
        ).pow(2).mean()

    # Train loop

    def pretty_print(*values):
        col_width = 13

        def format_val(v):
            if not isinstance(v, str):
                v = np.array2string(v, precision=6, floatmode='fixed')
            return v.ljust(col_width)

        str_values = [format_val(v) for v in values]
        print("   ".join(str_values))

    optimizer = optim.Adam(mlp.parameters(), lr=flags.lr)

    pretty_print(
        'step', 'train nll', 'train1 acc', 'train2 acc', 'irm penalty', 'rex penalty', 'iga penalty', 'fishr penalty', 'idm penalty', 'test acc'
    )
    for step in range(flags.steps):
        for edx, env in enumerate(envs):
            features, logits = mlp(env['images'])
            env['nll'] = mean_nll(logits, env['labels'])
            env['acc'] = mean_accuracy(logits, env['labels'])
            env['irm'] = compute_irm_penalty(logits, env['labels'])
            if edx in [0, 1]:
                # True when the dataset is in training
                optimizer.zero_grad()
                env["grads_variance"] = compute_grads_variance(features, env['labels'], mlp.classifier)
                optimizer.zero_grad()
                env["grads_sorted"] = compute_sorted_grads(features, env['labels'], mlp.classifier)

        train_nll = torch.stack([envs[0]['nll'], envs[1]['nll']]).mean()
        train_acc = torch.stack([envs[0]['acc'], envs[1]['acc']]).mean()

        weight_norm = torch.tensor(0.).cuda()
        for w in mlp.parameters():
            weight_norm += w.norm().pow(2)

        loss = train_nll.clone()
        loss += flags.l2_regularizer_weight * weight_norm

        irm_penalty = torch.stack([envs[0]['irm'], envs[1]['irm']]).mean()
        rex_penalty = (envs[0]['nll'].mean() - envs[1]['nll'].mean())**2

        iga_penalty = l2_between_grads_expectation(envs[0]["grads_sorted"], envs[1]["grads_sorted"])

        # Compute the variance averaged over the two training domains
        dict_grads_variance_averaged = OrderedDict(
            [
                (
                    name,
                    torch.stack([envs[0]["grads_variance"][name], envs[1]["grads_variance"][name]],
                                dim=0).mean(dim=0)
                ) for name in envs[0]["grads_variance"]
            ]
        )
        fishr_penalty = (
            l2_between_grads_variance(envs[0]["grads_variance"], dict_grads_variance_averaged) +
            l2_between_grads_variance(envs[1]["grads_variance"], dict_grads_variance_averaged)
        )

        idm_penalty = per_sample_distribution_matching(envs[0]["grads_sorted"], envs[1]["grads_sorted"])

        train_penalty, train_penalty2 = 0, 0
        # apply the selected regularization
        if flags.algorithm == "erm":
            pass
        else:
            if flags.algorithm == "irm":
                train_penalty = irm_penalty
            elif flags.algorithm == "rex":
                train_penalty = rex_penalty
            elif flags.algorithm == "iga":
                train_penalty = iga_penalty
            elif flags.algorithm == "fishr":
                train_penalty = fishr_penalty
            elif flags.algorithm == "idm":
                train_penalty = idm_penalty
            else:
                raise ValueError(flags.algorithm)
            penalty_weight = (flags.penalty_weight if step >= flags.penalty_anneal_iters else 1.0)
            loss += penalty_weight * train_penalty
            if penalty_weight > 1.0:
                # Rescale the entire loss to keep gradients in a reasonable range
                loss /= penalty_weight

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        test_acc = envs[2]['acc']
        grayscale_test_acc = envs[3]['acc']
        if step % 100 == 0:
            pretty_print(
                np.int32(step),
                train_nll.detach().cpu().numpy(),
                envs[0]['acc'].detach().cpu().numpy(),
                envs[1]['acc'].detach().cpu().numpy(),
                irm_penalty.detach().cpu().numpy(),
                rex_penalty.detach().cpu().numpy(),
                iga_penalty.detach().cpu().numpy(),
                fishr_penalty.detach().cpu().numpy(),
                idm_penalty.detach().cpu().numpy(),
                test_acc.detach().cpu().numpy(),
            )

    final_train_accs.append(train_acc.detach().cpu().numpy())
    final_test_accs.append(test_acc.detach().cpu().numpy())
    final_graytest_accs.append(grayscale_test_acc.detach().cpu().numpy())
    print('Final train acc (mean/std across restarts so far):')
    print(np.mean(final_train_accs), np.std(final_train_accs))
    print('Final test acc (mean/std across restarts so far):')
    print(np.mean(final_test_accs), np.std(final_test_accs))
    print('Final gray test acc (mean/std across restarts so far):')
    print(np.mean(final_graytest_accs), np.std(final_graytest_accs))
