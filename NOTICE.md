# Attribution

MEL was developed on top of NeuroBOLT, by Yamin Li, Ange Lou, Ziyuan Xu,
Shengchao Zhang, Shiyu Wang, Dario J. Englot, Soheil Kolouri, Daniel Moyer,
Roza G. Bayrak and Catie Chang (NeurIPS 2024).

Upstream code: https://github.com/soupeeli/NeuroBOLT

We thank the NeuroBOLT authors for openly sharing their code and resting-state
EEG-fMRI dataset, which support the data-processing foundation of this work.

The VU EEG/fMRI loading and preprocessing routines in data/loader.py and
data/preprocessing.py are adapted from that project. The existing GPLv3 license
is retained in LICENSE. No ownership of third-party datasets, atlases or
dependencies is asserted. Raw data and trained weights are not redistributed.

On 2026-10-15--2026-10-16, the curated release was restricted to the main matched-readout
experiment. Shared feature and readout definitions were extracted into
model/mel.py, the data-loading dependencies were reduced, and optional backbone
prediction copying was removed. Nonessential comments/docstrings were removed;
interpreter headers and licensing material remain. The release records the
verified historical target order without changing the fitted target ordering.

CANOE informed the repository presentation only. No CANOE algorithm, training
configuration, dataset, result or code was copied. This remains a GPLv3 project,
not a conversion to the license declared by that reference repository.
