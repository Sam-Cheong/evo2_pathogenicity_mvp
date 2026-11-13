"use strict";

const byId = (id) => document.getElementById(id);
const form = byId("composer");
const input = byId("message");
const send = byId("send");
const pending = byId("pending");
const messages = byId("messages");
let currentVariant;

function element(tag, value, className) {
  const node = document.createElement(tag);
  node.textContent = value;
  if (className) node.className = className;
  return node;
}

function scoreText(value) {
  return Number.isFinite(value) ? (100 * value).toFixed(1) + "%" : "—";
}

function signedDelta(value) {
  if (!Number.isFinite(value)) return "—";
  const magnitude = (100 * Math.abs(value)).toFixed(1);
  const sign = Number(magnitude) === 0 ? "" : value < 0 ? "−" : "+";
  return sign + magnitude + " pp";
}

function showEvidence(evidence) {
  const variant = evidence?.variant;
  if (!variant) return null;
  const details = element("details", "", "evidence");
  details.open = true;
  const hasPathogenicity = Number.isFinite(evidence.pathogenicity?.score_out_of_100)
    || Number.isFinite(evidence.pathogenicity?.score);
  const hasSplice = Array.isArray(evidence.disruptions) && evidence.disruptions.length > 0;
  details.append(element("summary", hasPathogenicity ? "Variant model evidence" : "Splice probe evidence"));
  const meta = element("div", "", "variant-meta");
  for (const value of [variant.gene, variant.variant_id, variant.hgvs_c || variant.hgvs_g || variant.hgvs_p]) {
    if (value) meta.append(element("span", value));
  }
  if (/^\d+$/.test(String(variant.clinvar_variation_id))) {
    const link = element("a", "View ClinVar ↗");
    link.href = "https://www.ncbi.nlm.nih.gov/clinvar/variation/" + variant.clinvar_variation_id + "/";
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    meta.append(link);
  }
  details.append(meta);
  const context = evidence.public_context;
  if (context) {
    details.append(element("p", "ClinVar: " + context.condition + " · " +
      context.clinvar_classification + " · " + context.hgvs_c, "evidence-note"));
    details.append(element("p", context.description, "evidence-note"));
    const guide = element("a", "About this condition (MedlinePlus) ↗");
    guide.href = "https://medlineplus.gov/genetics/condition/familial-hypercholesterolemia/";
    guide.target = "_blank";
    guide.rel = "noopener noreferrer";
    const line = element("p", "", "evidence-note");
    line.append(guide);
    details.append(line);
  }
  const selection = evidence.selection;
  if (selection?.matched_difference) {
    details.append(element("p", "Found " + selection.differences_found +
      " differences. Scored difference " + selection.matched_difference +
      ", the first exact ClinVar match.", "evidence-note"));
  }
  if (hasPathogenicity) {
    const {score, score_out_of_100: scaled, feature_source: source} = evidence.pathogenicity;
    const value = Number.isFinite(scaled) ? scaled : score * 100;
    const scoreCard = element("div", "", "pathogenicity");
    scoreCard.append(
      element("span", "Pathogenicity model score"),
      element("strong", value.toFixed(1) + " / 100"),
      element("p", "Research ranking score. Clinical probability is unavailable from this model.", "score-note")
    );
    if (typeof source === "string" && source) {
      scoreCard.append(element("p", "Feature source · " + source.replaceAll("_", " "), "score-note"));
    }
    details.append(scoreCard);
  }
  const drop = evidence.summary?.largest_drop_points;
  if (Number.isFinite(drop)) {
    details.append(element("p", "Largest splice-score drop: " + drop.toFixed(1) + " percentage points.", "evidence-note"));
  }
  if (hasSplice) {
    const table = element("table", "", "scores");
    const head = document.createElement("thead");
    const headers = document.createElement("tr");
    for (const value of ["Site", "Reference", "Variant", "Change"]) {
      const th = element("th", value);
      th.scope = "col";
      headers.append(th);
    }
    head.append(headers);
    const body = document.createElement("tbody");
    for (const row of evidence.disruptions) {
      const item = document.createElement("tr");
      const site = (row.id ? "[" + row.id + "] " : "") + row.annotation + " · " + row.position + " (" + row.strand + ")";
      const direction = row.delta < 0 ? "down" : row.delta > 0 ? "up" : "";
      item.append(element("td", site), element("td", scoreText(row.ref_score)),
        element("td", scoreText(row.alt_score)), element("td", signedDelta(row.delta), direction));
      body.append(item);
    }
    table.append(head, body);
    details.append(table);
  }
  if (hasSplice && evidence.evaluation?.validated !== true) {
    details.append(element("p", "These research scores describe possible changes in splicing. The chance that this variant causes disease is unavailable from this test.", "evidence-note"));
  }
  return details;
}

function showMessage(role, message, { evidence, model, error } = {}) {
  const row = element("li", "", "message " + role);
  row.append(element("p", role === "user" ? "You" : "Variant assistant", "speaker"));
  row.append(element("p", message, error ? "bubble error" : "bubble"));
  const panel = showEvidence(evidence);
  if (panel) row.append(panel);
  if (model) row.append(element("p", model, "model"));
  messages.append(row);
  return row;
}

async function requestReply(message) {
  let response;
  try {
    response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, variant_id: currentVariant, allow_nvidia: byId("allow-nvidia").checked }),
    });
  } catch {
    throw new Error("Could not connect to the local demo server.");
  }
  const result = await response.json().catch(() => ({})) || {};
  if (result.evidence?.variant?.variant_id) {
    currentVariant = result.evidence.variant.variant_id;
    byId("followup-sample").hidden = false;
  }
  if (!response.ok || typeof result.explanation !== "string" || !result.explanation.trim()) {
    return {
      message: typeof result.error === "string" ? result.error : "The server could not complete this request.",
      evidence: result.evidence,
      error: true,
      restoreInput: response.status === 422,
    };
  }
  return { message: result.explanation, evidence: result.evidence, model: result.model };
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message || send.disabled) return;
  byId("welcome").hidden = true;
  showMessage("user", message);
  input.value = "";
  send.disabled = true;
  byId("allow-nvidia").disabled = true;
  pending.hidden = false;
  pending.scrollIntoView({ block: "nearest" });
  let reply;
  try {
    reply = await requestReply(message);
  } catch (error) {
    reply = { message: error.message, error: true };
  } finally {
    pending.hidden = true;
    send.disabled = false;
    byId("allow-nvidia").disabled = false;
  }
  showMessage("assistant", reply.message, reply).scrollIntoView({ block: "start" });
  if (reply.restoreInput && !input.value) input.value = message;
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    form.requestSubmit();
  }
});

const promptSamples = {
  "case": "19-11110772-G-A\nExplain the splice evidence in plain language.",
  "single": "GRCh38 chr19:11110700\nDNA: ATGACCTTAAGATCGGCTACGAGTGCCTGTGCCCCGACGGCTTCCAGCTGGTGGCCCAGCGAAGATGCGAAGATGATTCCCGGGTGGGACTGAGCCCTGG\nExplain the research score in plain language.",
  "multiple": "GRCh38 chr19:11110700\nDNA: CTGACCTTAAGATCGGCTACGAGTGCCTGTGCCCCGACGGCTTCCAGCTGGTGGCCCAGCGAAGATGCGAAGATGATTCCCGGGTGGGACTGAGCCCTGG\nExplain the research score in plain language.",
  "window": "GCCAAGGTTGGCGGCGAAGGGATGGGTAGGGGCCCGAGAGTGACCAGTCTGCATCCCCTGGCCCTGCGCAGGGACCAACGAATGCTTGGACAACAACGGCGGCTGTTCCCACGTCTGCAATGACCTTAAGATCGGCTACGAGTGCCTGTGCCCCGACGGCTTCCAGCTGGTGGCCCAGCGAAGATGCGAAGATGATTCCCGGGTGGGACTGAGCCCTGGGCCCCCTCTGCGCTTCCTGACATGGCAACCAAACCCC\nExplain the splice evidence in plain language.",
  "fasta": ">LDLR_ALT\nGCCAAGGTTGGCGGCGAAGGGATGGGTAGGGGCCCGAGAGTGACCAGTCTGCATCCCCTGGCCC\nTGCGCAGGGACCAACGAATGCTTGGACAACAACGGCGGCTGTTCCCACGTCTGCAATGACCTTA\nAGATCGGCTACGAGTGCCTGTGCCCCGACGGCTTCCAGCTGGTGGCCCAGCGAAGATGCGAAGA\nTGATTCCCGGGTGGGACTGAGCCCTGGGCCCCCTCTGCGCTTCCTGACATGGCAACCAAACCCC\nExplain the splice evidence in plain language.",
  "pair": "REF: GCCAAGGTTGGCGGCGAAGGGATGGGTAGGGGCCCGAGAGTGACCAGTCTGCATCCCCTGGCCCTGCGCAGGGACCAACGAATGCTTGGACAACAACGGCGGCTGTTCCCACGTCTGCAATGACCTTAAGATCGGCTACGAGTGCCTGTGCCCCGACGGCTTCCAGCTGGTGGCCCAGCGAAGATGCGAAGGTGATTCCCGGGTGGGACTGAGCCCTGGGCCCCCTCTGCGCTTCCTGACATGGCAACCAAACCCC\nALT: GCCAAGGTTGGCGGCGAAGGGATGGGTAGGGGCCCGAGAGTGACCAGTCTGCATCCCCTGGCCCTGCGCAGGGACCAACGAATGCTTGGACAACAACGGCGGCTGTTCCCACGTCTGCAATGACCTTAAGATCGGCTACGAGTGCCTGTGCCCCGACGGCTTCCAGCTGGTGGCCCAGCGAAGATGCGAAGATGATTCCCGGGTGGGACTGAGCCCTGGGCCCCCTCTGCGCTTCCTGACATGGCAACCAAACCCC\nExplain the splice evidence in plain language.",
  "followup": "What does this result mean for someone reading it for the first time?"
};

document.querySelectorAll("[data-sample]").forEach((button) => {
  button.addEventListener("click", () => {
    const sample = button.dataset.sample;
    input.value = sample === "followup" && !currentVariant
      ? "19-11110772-G-A\n" + promptSamples.followup : promptSamples[sample];
    input.focus();
  });
});
