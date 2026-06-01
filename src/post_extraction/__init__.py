"""Post-extraction pipeline: R2 download -> merge -> split -> CLIP PCA -> normalise.

Operates on a local mirror of r2:clipwhy-data/{features,clip_embeddings,labeled}/.
Never writes back to R2. All outputs go to data/post_extraction/.
"""
