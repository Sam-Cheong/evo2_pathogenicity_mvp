# GRCh38 splice interpretation pilot

The workflow fits a position-level annotation probe for donor, acceptor and
background on frozen Evo 2 7B block-27 reference activations. It compares
the REF and ALT scores at one annotated splice anchor per public ClinVar SNV.
The annotation probe is separate from the existing pathogenicity classifier.

For each strand, the scored token is one base downstream of the intronic
anchor in transcript orientation. This allows both canonical donor bases
to enter the causal receptive field. Reported coordinates are 1-based
GRCh38 anchor coordinates, rather than the scored token coordinates.
The REF and ALT inputs use the same strand and 256-bp window.

## Prepare and train

Run from the project root in PowerShell. The annotation input in this example
is the [UCSC GENCODE Basic v49 genePred table](https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/wgEncodeGencodeBasicV49.txt.gz).
A matching GENCODE GTF is also supported. Use the repository's public ClinVar
GRCh38 VCF and indexed GRCh38 FASTA.

~~~powershell
$run = 'outputs/splice_pilot'
.\.venv\Scripts\python.exe -m evo2_mvp.splice_workflow prepare --gtf data/wgEncodeGencodeBasicV49.txt.gz --cohort "$run/cohort.json"
.\.venv\Scripts\python.exe -m evo2_mvp.splice_workflow fit --cohort "$run/cohort.json" --output $run
~~~

The fit stage uses the repository's NVIDIA API configuration for feature
extraction. Re-running fit in the same output directory reuses cached allele
vectors. It writes the fitted scaler and logistic probe to
~~~text
outputs/splice_pilot/splice_probe.json
~~~
and saves held-out results to metrics.json. The ClinVar challenge compares
the decrease in the expected donor or acceptor score at the supplied anchor.
The challenge compares canonical splice pathogenic variants with nearby
noncanonical benign controls, so it mixes allele effects with differences
in variant position and consequence. Its score is distinct
from the original covariance head's pathogenicity score.

## Score and generate a reply

Set OPENAI_API_KEY in the environment or in the repository's local .env
before requesting an explanation. Select a Responses API model available to
your OpenAI project. This one-command interface prints a JSON object that a
workflow can parse. The optional explanation field contains the generated
reply. Use a variant ID listed in the prepared cohort.

~~~powershell
$case = (Get-Content "$run/cohort.json" -Raw | ConvertFrom-Json).variants[0].variant_id
.\.venv\Scripts\python.exe -m evo2_mvp.splice_workflow score --cohort "$run/cohort.json" --output $run --variant-id $case --model "<your OpenAI model ID>"
~~~

Omit --model to return evidence without OpenAI, or call explain on an
existing evidence file to generate text without running Evo 2 again.

~~~powershell
.\.venv\Scripts\python.exe -m evo2_mvp.splice_workflow explain --evidence "$run/evidence/$case.json" --model "<your OpenAI model ID>"
~~~

The OpenAI request contains public variant identifiers, available metadata,
and ranked annotation scores. No sequence windows or embeddings are sent
to OpenAI. This pilot scores two annotation types at one anchor, so the
explanation has at most two evidence entries. The pilot keeps the genomic
HGVS from ClinVar. Coding and protein HGVS are optional and absent from
this VCF-derived cohort. Treat explanations as research
hypotheses. The probe supplies no clinical classification.
