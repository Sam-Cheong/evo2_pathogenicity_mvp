Sources: https://huggingface.co/LongSafari/hyenadna-tiny-1k-seqlen-d256-hf
Revision: 6036d4e144922b44470294f510cbfd2539ba3b7e

configuration_hyena.py, modeling_hyena.py and tokenization_hyena.py are
unmodified copies of the pinned checkpoint's implementation. MODEL_CARD.md
preserves the original model card, which declares BSD-3-Clause. The checkpoint
repository contains no standalone license text. LICENSE.upstream.txt preserves
the upstream HazyResearch/hyena-dna repository's Apache-2.0 license; that code
license and the checkpoint model-card declaration have different scopes.

Downloaded source SHA256:
configuration_hyena.py 29cb6cae67f5010e279d67349e688663043f0bd6791a09c36d96da8cac3668a6
modeling_hyena.py d78029cacea2c9259f1293fdb50b60f90872089469185e27d5c769086c475607
tokenization_hyena.py d424da2f794958b0360e033bbf5edc9ee4e3e3126e883093621348f07761e408

Project adapter evo2_mvp.hyenadna loads only verified local safetensors and
strictly validates all backbone keys, including the saved shared-frequency
aliases. Student inference reconstructs this bundled implementation without
trust_remote_code, a Hub lookup, pretrained-file access, or Evo 2.
