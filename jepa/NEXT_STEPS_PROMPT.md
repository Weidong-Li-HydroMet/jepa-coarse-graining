# Prompt — finish the spatiotemporal MJO V-JEPA (remaining work)

Paste this into a fresh Claude Code session on Glade to complete the parts that were left
running / unfinished. The implementation is already built and validated; what remains is the
**full GPU training run**, the **downstream phase/propagation verification on a properly
trained model**, and **executing the setup notebook**.

---

## Where things stand (already done — do NOT rebuild)
Project root: `/glade/campaign/univ/uchi0018/weidong/SSL_project/Self_supervised_learning/MJO`
Python env:  `/glade/work/weidong/.conda_envs/2DTurbEmulator/bin/python` (torch 2.9, timm, xarray, sklearn)

A self-contained spatiotemporal V-JEPA lives in `jepa/`:
- `jepa/data/` (anomaly.py = WH04 anomaly reused verbatim from `0-compute_rmm.ipynb`, kept 2-D;
  preprocess.py; dataset.py), `jepa/models/` (periodic-longitude ViT encoder, EMA target,
  narrow predictor, vjepa.py), `jepa/masks/` (longitude-sector tube + future-day masks),
  `jepa/downstream/` (extract.py, cluster.py, propagation.py).
- `jepa/config_vjepa.yaml`, `jepa/train_vjepa.py` (single-GPU; AMP, EMA, cosine LR, ckpt, resume,
  `--smoke`, `--epochs N`), `jepa/0-jepa_problem_setup.ipynb`, `jepa/README.md`.
- Read `jepa/README.md` first for the full layout and design decisions.

Validated:
- CPU smoke test passes (loss falls, no collapse); 7 pytest contracts pass
  (`jepa/tests/test_vjepa.py`); periodic longitude embedding and both masks verified.
- **Anomaly cache already built**: `data/anom_st/{anom.npy,dates.npy,stats.npz}` — 15,222 days,
  1979-04-30 → 2020-12-31, grid 13 lat × 144 lon (15°N–15°S, 2.5°). Do not rebuild.
- A short 8-epoch CPU "verify" job (`jepa/_verify.pbs`, job was 6443564) trained on real data
  with loss val 0.16→0.067→… and `tgt_std`≈0.37 (no collapse), writing `runs/vjepa/best.tar`.
  It may have hit its 1-h walltime during downstream — check `runs/vjepa/verify.pbs.log`.

## Hard constraints (important)
- **The login node has a 10 GB per-session memory cgroup shared with the user's VS Code/Jupyter
  (~9.5 GB already used). Any preprocessing/training/extract on the login node is OOM-killed.
  Run all heavy steps via PBS.**
- PBS: account `UCHI0018`. Accessible queues: `main` (routes to cpu/gpu) and `gpu`.
  `gpudev`/`cpudev`/`casper` are **denied**. Derecho GPU nodes allocate **whole** (4×A100).
- CPU verify scheduled instantly on `main`; the `gpu` queue is backlogged (expect a wait).

---

## TASK 1 — Full single-GPU pretraining (main deliverable still pending)
Submit the full run (100 epochs) + downstream on one A100:
```
cd /glade/campaign/univ/uchi0018/weidong/SSL_project/Self_supervised_learning/MJO
qsub jepa/submit_vjepa.pbs        # -q main, 1 node 4xA100 (pins CUDA_VISIBLE_DEVICES=0), 12h
qstat -u $USER ; tail -f runs/vjepa/train.pbs.log
```
This trains, then runs extract → cluster → propagation on `runs/vjepa/best.tar`.
Monitor to completion. On an A100 each epoch should be seconds–minutes (CPU was ~390 s).
If the `gpu` queue won't schedule in reasonable time, you may instead run more CPU epochs via
`jepa/_verify.pbs` (bump `--epochs`, raise walltime), but GPU is strongly preferred.

## TASK 2 — Downstream verification on the trained model (the science checks)
After training, confirm / report (these are the acceptance criteria from the original spec):
1. **Phase clustering vs RMM** (`jepa.downstream.cluster`, k=8): report ARI / NMI / circular
   accuracy of the V-JEPA phases vs the classical WH04 RMM phase (active days), from
   `rmm_computed.csv`. Open `runs/vjepa/phase_composites.png` and confirm the k-means/SOM
   composites show **coherent eastward-propagating** OLR/U850 structure (enhanced convection —
   blue OLR — marching east as cluster id increases). Cluster assignment should correlate with
   the RMM phase of the same days (ARI/NMI clearly > 0; circular_acc well above chance 1/8).
   If agreement is weak, train longer / raise `output_config`-style effort (more epochs), or try
   `pool='all'` in extract, or k=8 SOM iters.
2. **Propagation classifier** (`jepa.downstream.propagation`): report held-out accuracy of the
   standing/jumping/slow/fast head vs the majority baseline; it should beat baseline if the
   representation captured propagation. Inspect the confusion matrix in `propagation.npz`.
3. **Collapse monitor**: confirm final `tgt_std` stays well above 0 in `train.pbs.log`.
   If it trends toward 0, set `method.var_coeff: 0.1` in `jepa/config_vjepa.yaml` and retrain.

## TASK 3 — Execute the setup notebook (embed outputs)
Run `jepa/0-jepa_problem_setup.ipynb` end-to-end so its figures (sample 3-day clip, masking
illustration, periodic-embedding curve, architecture summary, one forward pass) are saved in the
notebook. The cache now exists, so it will plot real data. Because of the login-node memory cap,
execute it on a compute node, e.g.:
```
qsub -I -A UCHI0018 -q main -l select=1:ncpus=16 -l walltime=00:30:00
# then, on the compute node:
/glade/work/weidong/.conda_envs/2DTurbEmulator/bin/jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=1200 --inplace jepa/0-jepa_problem_setup.ipynb
```
(`jupyter` may need `pip install nbconvert` in that env, or use `papermill` if present.)

## TASK 4 — Report
Summarize: final train/val latent loss + curve, collapse metric, phase-vs-RMM agreement numbers
(+ the composites figure), propagation accuracy vs baseline, and restate the assumptions
(2.5° grid, 15°N–15°S, WH04 anomalies, 3-day clips, train 1979–2014 / val 2015–2020, mask params).
Then, optionally, promote the V-JEPA to the default by renaming `train_vjepa.py`→`train.py` and
`config_vjepa.yaml`→`config.yaml` (the legacy 2-D I-JEPA comparison files currently hold those
names — see `jepa/README.md` "Relation to the existing jepa/ files").

## Gotchas to remember
- Don't run heavy Python on the login node (OOM). Use PBS (`main`/`gpu`, `UCHI0018`).
- `preprocess.py` is already done — skip unless `data/anom_st/anom.npy` is missing.
- `train_vjepa.py --epochs N` overrides config epochs; `--resume runs/vjepa/last.tar` resumes.
- Downstream scripts take `--ckpt runs/vjepa/best.tar` and write embeddings/labels/figures into
  `runs/vjepa/`.
