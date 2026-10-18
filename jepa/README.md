# Spatiotemporal MJO V-JEPA

A **Joint-Embedding Predictive Architecture** that learns a self-supervised
representation of the MJO from 3-day sequences of tropical daily anomaly maps
(OLR, U850, U200), so the per-day embeddings can be clustered into MJO **phases**
and used to classify MJO **propagation type** (standing / jumping / slow / fast).

Unlike a reconstruction autoencoder (PhysAnchor), JEPA predicts **in latent space**:
from the visible part of a clip it predicts the *representation* of a hidden part — a
masked longitude sector (translation-tolerance) or the next day (propagation). No pixel
reconstruction.

## Layout
```
config_vjepa.yaml         all paths + hyperparameters (no hardcoded paths in logic)
train_vjepa.py            single-GPU trainer (AMP, EMA, cosine LR, ckpt, resume, --smoke)
0-jepa_problem_setup.ipynb  Deliverable 1: setup, sample clips, masking, architecture
.build_jepa_problem_setup.py  regenerates the notebook

data/
  anomaly.py     WH04 anomaly ops copied verbatim from 0-compute_rmm.ipynb (2-D, no merid. mean)
  preprocess.py  build standardized 2-D anomaly cache  ->  data/anom_st/
  dataset.py     MJOSequence (3-day clips) + SyntheticSequence (smoke test)
models/
  pos_embed.py   periodic-longitude 3-D sin-cos embedding
  encoder.py     STViTBackbone — spatiotemporal ViT (reuses common.PatchEmbed)
  predictor.py   narrow-ViT predictor (context + positioned mask tokens -> target latents)
  vjepa.py       VJEPA: context enc + EMA target enc + predictor + latent loss + collapse monitor
masks/
  masking.py     spatial longitude-sector (tube, wrap) + temporal future-day masks
downstream/
  extract.py     one embedding per day (EMA target encoder, current-day pooling)
  cluster.py     SOM + k-means phases, ordered by convective longitude, RMM-phase agreement
  propagation.py MJO-event propagation-type labels + supervised head
reference/       cloned facebookresearch/{jepa, ijepa} (reference only)
```

## Environment
```
conda activate /glade/work/weidong/.conda_envs/2DTurbEmulator   # torch 2.9, timm, xarray, sklearn
```

## Run
```bash
cd /glade/campaign/univ/uchi0018/weidong/SSL_project/Self_supervised_learning/MJO

python -m jepa.data.preprocess --config jepa/config_vjepa.yaml      # once (needs xarray)
python jepa/train_vjepa.py --config jepa/config_vjepa.yaml --smoke  # CPU smoke test
python jepa/train_vjepa.py --config jepa/config_vjepa.yaml          # single GPU

python -m jepa.downstream.extract     --ckpt runs/vjepa/best.tar
python -m jepa.downstream.cluster     --ckpt runs/vjepa/best.tar
python -m jepa.downstream.propagation --ckpt runs/vjepa/best.tar
```

## Key decisions / assumptions
- **Anomaly definition reused verbatim** from `0-compute_rmm.ipynb` (`data/anomaly.py`):
  seasonal cycle = mean + 3 annual harmonics (LSQ over base period), then trailing-120-day
  mean removed; **no bandpass**. The RMM notebook meridionally averages; we keep the full
  2-D map.
- **Grid:** 2.5°, **15°N–15°S** band (13 lat × 144 lon), full periodic longitude — matches
  the RMM notebook, so the phase sanity-check against `rmm_computed.csv` is apples-to-apples.
- **Sample:** 3 consecutive days `(C=3, T=3, 13, 144)`; patch `(1, 8)` → 13×18 patches/day,
  702 tokens/clip; periodic-longitude 3-D positional embedding.
- **Masking:** spatial contiguous-longitude sector (tube across all days, wraps 0/360°) and
  temporal future-day; `mixed` by default. Context/target disjoint (I-JEPA convention).
- **Objective:** smooth-L1 in latent space vs the EMA target encoder (stop-grad). Anti-collapse =
  EMA + predictor asymmetry; `tgt_std` is logged; an optional VICReg variance hinge (`var_coeff`).
- **Split:** train 1979–2014, val 2015–2020, by anchor-day year; per-variable standardized on
  the training years.

## Relation to the existing `jepa/` files
`jepa/model.py`, `jepa/train.py`, `jepa/config.yaml` are the earlier **2-D I-JEPA**
member of the MAE/DINO/I-JEPA comparison (single daily map, shared `common/backbone_vit.py`,
fairness contract enforced by `scripts/smoke_test.py`). They are left untouched; the
spatiotemporal V-JEPA here is self-contained (`train_vjepa.py` + `config_vjepa.yaml` +
the `data/ models/ masks/ downstream/` subpackages). To make this the default, rename
`train_vjepa.py`/`config_vjepa.yaml` after retiring the comparison's references.
