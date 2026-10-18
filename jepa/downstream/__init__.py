"""Downstream use of the pretrained V-JEPA representation.

extract.py     -> one embedding per day (EMA target encoder, current-day pooling)
cluster.py     -> SOM / k-means phases, ordered by convective longitude, RMM agreement
propagation.py -> MJO-event propagation-type labels (standing/jumping/slow/fast) + head
"""
