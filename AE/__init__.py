"""PhysAnchor-MJO-AE — a reconstruction-autoencoder baseline matched to the V-JEPA.

Same ``STViTBackbone`` encoder and the same latent dimension as ``jepa`` (read from the
shared config), the same data/dataloader (``jepa.data.dataset``), and the same downstream
science (SOM/k-means phase anatomy + pattern-correlation monitoring) — only the model and
objective differ (plain pixel reconstruction instead of latent-space prediction), so the
two representations are an apples-to-apples comparison.
"""
