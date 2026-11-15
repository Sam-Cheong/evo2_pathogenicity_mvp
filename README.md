# Short-window pathogenicity prediction with Evo 2

This project classifies human ClinVar single-nucleotide variants (SNVs) using a small covariance head on frozen Evo 2 7B activations. It includes a local web demo that resolves an SNV against GRCh38, computes a research score, and explains the available evidence in plain language. The classifier is intended for research. Its score cannot estimate a person's chance of disease.

[Web demo](#web-demo) · [Model](#model) · [Results](#results) · [Reproduction](#reproduction)

## Web demo

The browser app has two routes:

- **Score a DNA fragment.** Enter a 1-based GRCh38 start coordinate and 20–2,000 forward-strand DNA bases. The server compares the fragment with the local reference. It scores a single difference directly. For multiple differences, it checks the local ClinVar index in ascending coordinate order and scores the first exact chromosome, position, REF, and ALT match.
- **Explore splice evidence.** Enter a supported ClinVar variant ID or paste an exact 256-bp window from the 24-case splice pilot. This route reports separate donor and acceptor score changes. The home page includes clickable examples for both routes.

For the DNA route, the server reconstructs four 256-bp windows, pairing REF and ALT in each orientation. It reuses verified activation features when available. On a cache miss, the NVIDIA checkbox authorizes a request for the reconstructed windows. The server runs the local PyTorch checkpoint on the resulting features for each score. The reconstructed ALT window contains the variant, so use public examples for demonstrations. Questions and numerical evidence can be sent through [chat-request.ps1](chat-request.ps1). The original DNA input is excluded from the chat prompt.

### Run on Windows

From the repository root, set up Python 3.11:

~~~powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
~~~

Place the matching, uncompressed reference FASTA at data/GRCh38.fa. The server checks its SHA-256 against the saved models. The expected reference checksum is recorded in [the cohort metadata](data/variants.metadata.json). The full FASTA and ClinVar VCF are local inputs that must be supplied separately.

For cache misses, configure NVIDIA_API_KEY in the local environment or project .env. For generated chat replies, use PowerShell 7 and set OPENAI_API_KEY in the project .env for chat-request.ps1. The browser can still display the model evidence when the explanation request fails.

For the multiple-difference route, place the GRCh38 ClinVar VCF at data/clinvar.vcf.gz and build its local index:

~~~powershell
.\.venv\Scripts\python.exe .\demo\splice_web\clinvar_index.py
~~~

Start the server and open [http://127.0.0.1:8765](http://127.0.0.1:8765):

~~~powershell
.\.venv\Scripts\python.exe .\demo\splice_web\server.py
~~~

The demo listens on this computer only. For more input examples and deployment details, see [the web demo guide](demo/splice_web/README.md).

## Model

Each SNV is represented by four 256-bp sequences, one REF and one ALT sequence for each of the forward and reverse-complement orientations. In each orientation, the SNV is at the 193rd position. The preceding 192 bases provide context, and the 64 candidate positions start at the SNV.

1. Retrieve per-base activations from **Evo 2 7B blocks.27** through the NVIDIA-hosted forward endpoint. Each input yields a 256 × 4096 array. The provider did not expose an exact checkpoint revision.
2. Within each orientation, rank the 64 candidate positions by REF–ALT cosine divergence and retain the top 32. Subtract REF activations from ALT activations at each selected position, producing a **2 × 32 × 4096** tensor per SNV.
3. Combine the 64 selected vectors into a matrix D. The covariance head maps D through two learned rank-16 projections, pools their uncentered cross-moment, and passes the resulting 16 × 16 matrix to a linear classifier.

The **131,329-parameter head** is trained on CPU from cached activations. A sigmoid maps its logit to a research score, which the demo displays on a 0–100 scale. The mean-pooling comparator uses the same selected positions and fits a logistic regression after averaging the features separately for each orientation.

[Classifier implementation](src/evo2_mvp/evee_covariance.py) · [Trained checkpoint](outputs/evee_full_2000_20261111/probe/evee_covariance.pt)

## Results

The ClinVar cohort contains **2,000 GRCh38 autosomal SNVs**, split by gene into 1,200 training, 400 validation, and 400 test variants. The test split includes 48 positives and 32 variants inspected during earlier exploration.

| Readout | Test AUROC | Test AUPRC (trapezoidal) | Test AP |
|---|---:|---:|---:|
| Covariance | **0.9612** | **0.8843** | **0.8852** |
| Mean pooling | 0.9453 | 0.8474 | 0.8487 |

The paired 95% gene-group bootstrap intervals for the covariance-minus-mean differences include zero for both AUROC and AUPRC. The Evo 2 7B native-likelihood result in the Evo 2 preprint's Table S7 was 0.960 AUROC and 0.931 AUPRC on a separate 49,281-SNV cohort. The cohort and scoring differences prevent a direct ranking.

The report also includes a separate author-measured RTX 3080 Laptop GPU comparison. The reduced-backbone probe pipeline took 208 ms per variant and 8.21 GiB of peak CUDA allocated memory, versus 319 ms and 12.43 GiB for full-model likelihood scoring. Both the backbone and scoring rule change between the two configurations.

[Evaluation artifact](outputs/evee_full_2000_20261111/probe/evee_covariance_metrics.json) · [Per-variant predictions](outputs/evee_full_2000_20261111/probe/evee_covariance_predictions.csv)

The evaluation artifact uses average precision in its auprc field. The table above reports AP separately from trapezoidal AUPRC.

## Reproduction

- [Short-window probe experiment](docs/EVEE_SHORT_WINDOW.md) and [full-cohort runner](scripts/run_evee_full.py)
- [Splice annotation and explanation workflow](docs/SPLICE_WORKFLOW.md)
- [Earlier 40B teacher and HyenaDNA student experiments](docs/HYENADNA_EXPERIMENT.md)
- [Earlier two-stage StripedHyena2 distillation](docs/SH2_TWO_STAGE_EXPERIMENT.md)
- [Hosted API setup and protocol audit](docs/nvidia_api_audit.md)
