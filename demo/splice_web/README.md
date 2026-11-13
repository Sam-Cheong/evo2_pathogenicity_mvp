# Splice interpretation chat demo

Run from the repository root in PowerShell.

    .\.venv\Scripts\python.exe .\demo\splice_web\server.py

Open http://127.0.0.1:8765. Use Ctrl+C to stop the server. If the port
is already in use, launch with --port 8766 and open that port instead.

Send a supported ClinVar case ID or paste an exact, transcript-oriented
256-base REF or ALT window from one of the 24 splice pilot cases. You can
paste a wrapped FASTA record or label separate blocks REF: and ALT:. Put
the question on its own line. Try:

    Explain the predicted splice changes for 19-11110772-G-A.

The server matches pasted sequences locally. It loads the saved probe and
runs it on cached REF and ALT Evo 2 embeddings, which were precomputed
using NVIDIA's hosted 7B service. The demo does not run Evo 2 during chat
and does not score novel sequences. The local GRCh38 reference, pilot
cohort, checkpoint and token cache must be present.

After scoring, the server passes the question and public variant metadata
and numerical evidence to chat-request.ps1. DNA blocks and FASTA headers
are excluded from the prompt. Configure the same local credentials that
make the script work in PowerShell. The browser receives evidence and
generated text. If generation fails, the local evidence remains visible.
The server listens on loopback and is intended for local presentation.
## Score a GRCh38 fragment with the covariance head

For the pathogenicity route, enter the 1-based start coordinate of a
forward-strand autosomal fragment on the first line. On the second line,
enter DNA: followed by 20 to 2,000 A/C/G/T bases. Put any question on a
separate line. The server compares the fragment with GRCh38 at that
coordinate and scores a single difference. For multiple differences it checks the local ClinVar VCF index in ascending genomic order and scores the first exact REF/ALT match. Supply the mapping
from the sequencing report. The demo does not search for a location across
the genome, call variants from raw reads, or handle indels or ambiguous bases.

The server reconstructs 256-bp REF and ALT windows for both orientations
and checks the original feature cache against the reconstructed sequence
hashes. An uncached SNV requires the NVIDIA option. Only those four
reconstructed windows are sent to NVIDIA Evo 2. The original pasted
fragment is excluded from the chat prompt. A reconstructed ALT window
still reveals the variant and its context to NVIDIA. Use public demo data
for presentations and keep personal information out of the question.

The local model is the trained evee_covariance.pt. It receives the two
sets of 32 position-wise ALT-minus-REF vectors and scores the variant
without reading the predictions CSV. The displayed research score is
100 times sigmoid of the model logit, rounded to one decimal place.
It is not a calibrated probability of disease. New feature vectors
are cached in memory for the current server session. If the chat
service is unavailable, the interface still returns the local score.

Build or refresh the local ClinVar index with:

    .\.venv\Scripts\python.exe .\demo\splice_web\clinvar_index.py

## Launch and open the demo

Run .\demo\splice_web\start-web.ps1 from the repository root to start the
server in the background and open the default browser when it is ready.
An existing instance on the requested port is reused. Use -Port 8767 to
select another port or -NoBrowser to print the URL without opening it.
