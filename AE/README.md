# PhysAnchor-MJO-AE — reconstruction-autoencoder baseline (matched to V-JEPA)

Implementation of the autoencoder method of **arXiv:2512.22144** ("The Complete Anatomy of
the MJO Revealed by AI"), built as a **like-for-like baseline for the V-JEPA** in `../jepa`.
Same encoder, same latent dim, same data — only the objective differs (pixel reconstruction
instead of latent prediction), so the two learned representations are an apples-to-apples
comparison.

## The two hard constraints (vs the paper)

| | paper | here (matched to V-JEPA) |
|---|---|---|
| encoder | plain 3-layer Transformer + CNN | **`jepa.models.encoder.STViTBackbone`** (identical class/arch) |
| latent dim | 1024 | **192** (`= encoder.embed_dim`, asserted at build) |
| data | 20–90 d Lanczos + wavenumber-1–3 filter | the V-JEPA **WH04 anomaly cache, verbatim** (`data/anom_st`) |

What is kept from the paper: 3-consecutive-day input (t−2, t−1, t), **reconstruct day t**,
**MSE** loss, no masking/EMA/predictor; downstream **SOM 4×2 → 6 canonical phases** (k-means
k=6 validation, ordered by convective-centre longitude), and the **decoupled
pattern-correlation monitoring index** (location + intensity) on the 30°E–210°E sector.

## Layout

```
AE/
  config_ae.yaml            data:/encoder: copied verbatim from jepa/config_vjepa.yaml; latent_dim pinned to 192
  models/decoder.py         SpatialDecoder: latent (B,192) -> day-t field (B,3,13,144)
  models/ae.py              MJOAutoencoder: shared encoder + pool + decoder + MSE (asserts latent==embed_dim)
  train_ae.py               single-GPU train (AMP, cosine LR, best/last, resume, --smoke)
  submit_ae.pbs             Derecho A100 job, mirrors jepa/submit_vjepa.pbs
  downstream/extract.py     one latent per day (current-day pool, same as V-JEPA)
  downstream/phases.py      SOM 4x2 -> select 6 (wavenumber-1) -> k-means k=6 -> order -> composites -> RMM
  downstream/monitor.py     decoupled location/intensity index; spurious-propagation & misplacement vs RMM
  downstream/compare.py     AE vs V-JEPA: silhouette, RMM ARI/NMI, composite coherence -> table + figure
  0-ae_problem_setup.ipynb  setup notebook (SI_PlaSim style)
```

## Run

```bash
cd /glade/campaign/univ/uchi0018/weidong/SSL_project/Self_supervised_learning/MJO
python AE/train_ae.py --config AE/config_ae.yaml --smoke   # CPU smoke test (asserts loss falls)
qsub AE/submit_ae.pbs                                       # full single-GPU train + all downstream
```

The full job trains, then runs `extract → phases → monitor → compare` on `runs/ae/best.tar`
(for the AE the reconstruction val loss IS a valid model-selection signal, unlike latent-space
V-JEPA which uses `last.tar`).

## Notes / documented choices

- **Local sector.** The encoder sees the full global periodic field (so its input samples are
  byte-identical to V-JEPA's — hard constraint #3); the paper's 30°E–210°E sector restriction
  is applied where the paper applies it, in the downstream pattern-correlation monitor. The
  3-day window itself provides the propagation-locality (~5°/day → <15° over the clip).
- **Canonical-phase selection** uses the zonal-wavenumber-1 OLR power fraction as an objective
  proxy for the paper's "coherent planetary wavenumber-1" criterion (keeps ~75% of days).
- **Intensity** `I_MJO = α·sqrt(Σ wᵢ αᵢ²)`, α=3 — the paper's weighted-RMS wording on the RMM scale.
```
