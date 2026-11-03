# 公开真实数据管线演示（不是完整训练队列）

本目录是本次开发实际下载并校验的数据，**没有生成 Teacher 或 Student 预测，也没有训练性能结果**。

- 官方 ClinVar GRCh38 VCF：`https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz`
- VCF 内部发布日期：2026-11-01。本次仅通过流式读取保存前 **5,000 条数据记录**，范围 chr1:66926–1211584；这不是完整 ClinVar 版本。上游压缩文件响应 Content-Length 为 201,346,922 字节，未完整下载。
- 官方参考序列：NCBI RefSeq **NC_000001.11**，GRCh38.p14 染色体 1 的 **1–1,212,000** 区段。FASTA 第一行规范化为 `>1`，序列碱基及坐标未变。该文件只适合本前缀范围；不是完整 GRCh38 FASTA。
- 使用严格 P/LP、B/LB 标签、至少两星审核、常染色体 SNV、去重以及 256 bp 完整上下文与 REF 核验后，实际保留 **473 条**，其中 **466 条 B/LB，7 条 P/LP**；没有因 REF 不匹配或上下文不合法而丢失候选。
- 目标 2,000 条未达到，没有放宽标签条件。由于下载范围和类别严重偏倚，该演示队列不适合声称泛化性能。正式研究应使用完整官方版本及完整 GRCh38 FASTA，重新筛选及按基因划分。

文件：`clinvar_official_prefix.vcf` 为原始流式前缀；`grch38_chr1_prefix.fa` 为上述局部参考；`variants.csv` 为实际筛选结果；`variants.metadata.json` 为筛选统计及文件 SHA-256；`download_provenance.json` 为下载范围、来源、日期和校验值。CSV 中每行 `source=ClinVar, synthetic=false`。

可在项目根目录重跑筛选：

```bash
python scripts/prepare_clinvar.py --vcf data/real_demo/clinvar_official_prefix.vcf --fasta data/real_demo/grch38_chr1_prefix.fa --out outputs/real_demo_rechecked.csv --release-date 2026-11-01 --target 2000 --window 256
```

FASTA 的局部来源请求：

`https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_000001.11&seq_start=1&seq_stop=1212000&rettype=fasta&retmode=text`
