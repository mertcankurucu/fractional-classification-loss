
# Fractional Classification Loss (FCL)

> 🚧 **Under construction**
>
> This repository currently contains documentation only. The PyTorch implementation, examples, and tests will be made publicly available upon acceptance of the paper. 

**Adaptive robust loss function for classification with noisy labels. The robustness level automatically adjusts during training, without manual robustness parameter tuning.**

PyTorch implementation of **Fractional Classification Loss (FCL)**, introduced in:

> M. C. Kurucu, T. Kumbasar, İ. Eksin, M. Güzelkaya. *Introducing Fractional Classification Loss for Robust Learning with Noisy Labels.* [arXiv:2508.06346](https://arxiv.org/abs/2508.06346), 2025.

## Why FCL?

Training classifiers with noisy labels requires balancing **fast convergence** and **robustness to incorrect labels**. Cross-entropy (CE) converges quickly but is sensitive to label noise, while mean absolute error (MAE) is robust but often suffers from slow convergence and underfitting. Existing robust losses typically require manually tuning robustness parameters for different datasets.

**FCL automatically learns the appropriate robustness level during training.**

FCL combines fractional cross-entropy with MAE and introduces a single learnable fractional order **μ** that controls the robustness–convergence trade-off. Rather than fixing this parameter beforehand, FCL optimizes it jointly with the classifier.

- **Simple integration:** Works with standard PyTorch classifiers without auxiliary networks or sample-selection procedures. Only one additional call, `criterion.step_mu()`, is required per epoch.

## Installation

```bash
git clone https://github.com/mertcankurucu/fractional-classification-loss.git
cd fractional-classification-loss
pip install -e .            # needs PyTorch; the examples also need torchvision
```

## Quick start

```python
import torch
from fcl import FCLLoss

model = ...                                     # any classifier that outputs logits
criterion = FCLLoss(mu_init=0.5, mu_lr=0.1, warmup_epochs=5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)      # network parameters only

for epoch in range(num_epochs):
    for x, y in train_loader:
        loss = criterion(model(x.to(device)), y.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    criterion.step_mu()                         # update the fractional order once per epoch
    print(epoch, criterion.mu)
```

That is the whole integration: call `criterion.step_mu()` once at the end of every epoch. Keep the loss's own parameter (μ′) out of the network optimizer, weight decay and gradient clipping; `FCLLoss` updates it itself.

To resume training, save these three things alongside the model:
- `criterion.state_dict()`, which contains μ′;
- `criterion.mu_optimizer.state_dict()`;
- `criterion.epoch`.

## Baseline losses

All losses share the interface `loss = criterion(logits, targets)`. Build them with `fcl.get_loss(name, **params)`, or import the classes directly.

| Name | Class | Parameters | Reference |
|---|---|---|---|
| `fcl` | `FCLLoss` | `mu_init`, `mu_lr`, `warmup_epochs` | this paper |
| `ce` | `CELoss` | – | cross-entropy |
| `mae` | `MAELoss` | – | Ghosh et al., AAAI 2017 |
| `gce` | `GCELoss` | `q` | Zhang & Sabuncu, NeurIPS 2018 |
| `sce` | `SCELoss` | `alpha`, `beta` | Wang et al., ICCV 2019 |
| `nce_mae` | `NCEandMAE` | `alpha`, `beta` | Ma et al., ICML 2020 |
| `nce_rce` | `NCEandRCE` | `alpha`, `beta` | Ma et al., ICML 2020 |
| `anl_ce` | `ANLCELoss` | `alpha`, `beta`, `delta`* | Ye et al., NeurIPS 2023 |
| `mae_star` | `MAEStarLoss` | `eps` | Staats et al., AAAI 2025 |

\* ANL-CE's L1 penalty `delta * Σ|θ|` on the network weights is added in the training loop; see `examples/train.py`.

The formulas are in [`fcl/losses.py`](fcl/losses.py). In this code, MAE is $\sum_k |p_k - y_k| = 2(1-p_y)$, and RCE clips the one-hot label at $10^{-4}$.

## Examples: MNIST, CIFAR-10 and CIFAR-100 with symmetric label noise

`examples/train.py` trains the networks used in the paper:
- a 2-block CNN on MNIST;
- an 8-layer CNN on CIFAR-10;
- a ResNet-34 on CIFAR-100.

Symmetric noise replaces each training label, with probability `--rate`, by one of the other classes chosen uniformly.

```bash
python examples/train.py --dataset mnist    --rate 0.8
python examples/train.py --dataset cifar10  --rate 0.6
python examples/train.py --dataset cifar100 --rate 0.4
python examples/train.py --dataset cifar10  --rate 0.6 --loss nce_rce      # a baseline, for comparison
```

| Dataset | Network | Epochs | Weight decay | FCL: μ₀ / μ LR / warm-up |
|---|---|---|---|---|
| MNIST | 2-block CNN | 50 | 1e-3 | 0.5 / 0.2 / 0 |
| CIFAR-10 | 8-layer CNN | 120 | 1e-4 | 0.5 / 0.1 / 5 |
| CIFAR-100 | ResNet-34 | 200 | 1e-5 | 0.5 / 0.1 / 5 |

Common settings:
- **Optimizer:** Adam (lr 1e-3) with per-iteration cosine decay, batch size 256.
- **Gradients:** values clipped to ±10.
- **Augmentation:** horizontal flip (MNIST); random crop and flip (CIFAR).
- **Downloads:** datasets go to `./data`.
- **Overrides:** `--mu_init`, `--mu_lr`, `--warmup_epochs`, `--epochs`.

The baseline parameters used for each dataset are in `LOSS_ARGS` at the top of the script.

This is the PyTorch implementation. The paper's MNIST and CIFAR tables were produced with the authors' original MATLAB code, so numbers obtained with this code can differ slightly.

## Tests

```bash
pip install pytest
python -m pytest -q tests
```

## Citation

```bibtex
@article{kurucu2025fcl,
  title   = {Introducing Fractional Classification Loss for Robust Learning with Noisy Labels},
  author  = {Kurucu, Mert Can and Kumbasar, Tufan and Eksin, {\.I}brahim and G{\"u}zelkaya, M{\"u}jde},
  journal = {arXiv preprint arXiv:2508.06346},
  year    = {2025}
}
```

## License

[MIT](LICENSE)\
