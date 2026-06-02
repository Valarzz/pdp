# From Noise to Control: Parameterized Diffusion Policies

By [Renhao Zhang](https://renhaoz.github.io/), [Haotian Fu](https://haotianfu.me/), [Mingxi Jia](https://mingxi-jia.github.io/), [George Konidaris](https://cs.brown.edu/people/gdk/), [Yilun Du](https://yilundu.github.io/), and [Bruno Castro da Silva](https://people.cs.umass.edu/~bsilva/).

> We propose Parameterized Diffusion Policy (PDP), a framework that learns a diffusion policy parameterized in a smooth continuous space. By structuring a latent manifold such that distances between latents' values reflect the semantic similarity of physical trajectories, we transform diffusion from a mechanism of stochastic diversity into a precise tool for behavior steering. Our approach also enables smooth interpolation between known strategies and efficient generalization to novel constraints without the need to update policy weights. We demonstrate that PDP significantly improves adaptation performance on complex multimodal benchmarks in both simulation and real-robot hardware compared to regular diffusion policy, particularly in scenarios requiring the discovery of novel behaviors.

## About the paper 

[Paper](https://arxiv.org/abs/2606.00336)

[Project Page](https://sites.google.com/view/parameterized-dp)

Presented at ICML 2026.

## Requirements

Python 3.10+. Install the core dependencies with `pip install -r requirements.txt`.

## Domain

We include four RLBench tasks: `close_drawer`, `place_block`, `meat_off_grill`, and `pick_up_cup`. Their pre-processed data (multimodal datasets, normalizers, and adaptation demos) is provided under `data/`. The simulator is not bundled here — install [RLBench](https://github.com/stepjam/RLBench) (with PyRep + CoppeliaSim) and follow their instructions for setup; it is only needed for the adaptation rollout.

## Usage

Run from the repo root. All four tasks share the same code; only the config differs — swap `close_drawer` below for `place_block`, `meat_off_grill`, or `pick_up_cup`.

### 1. Train the trajectory encoder

```bash
python train_encoder.py --config config/close_drawer/encoder.yaml
```

Saves the encoder to `results/close_drawer/encoder/best_model.pt`.

### 2. Train the parameterized diffusion policy

```bash
python train_pdp.py --config config/close_drawer/pdp.yaml
```

Loads the encoder from step 1 and saves the policy to `results/close_drawer/pdp/best_model.pt`.

### 3. Adapt to a shifted scene

Optimizes only the latent z from a single demo (no weight updates), then rolls the adapted policy out in RLBench. Each task ships three shifted-scene demos:

```bash
python adapt.py --config config/close_drawer/eval.yaml --demo shiftdemo1
python adapt.py --config config/close_drawer/eval.yaml --demo shiftdemo2
python adapt.py --config config/close_drawer/eval.yaml --demo shiftdemo3
```

The rollout runs through `make_rlbench_env` in `adapt.py`, a single integration point you implement against your RLBench install (see [Domain](#domain)). To run the adaptation alone — latent optimization and trajectory generation, no simulator — add `--no_rollout`:

```bash
python adapt.py --config config/close_drawer/eval.yaml --demo shiftdemo1 --no_rollout
```

## Citation

```bibtex
@misc{zhang2026noisecontrolparameterizeddiffusion,
      title={From Noise to Control: Parameterized Diffusion Policies}, 
      author={Renhao Zhang and Haotian Fu and Mingxi Jia and George Konidaris and Yilun Du and Bruno Castro da Silva},
      year={2026},
      eprint={2606.00336},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2606.00336}, 
}
```
