# Public-source eligibility search protocol

**Audit freeze:** 2026-08-07
**Audit identifier:** `MOPR-FQELIG1`
**Scope:** candidate acquisition for an independent action--quantum endpoint
test; not a systematic review of all quantum experiments.

## Discovery sources

The candidate search was iterative across arXiv, Crossref/DOI landing pages,
Zenodo, Dryad, Figshare and journal data-availability statements. The frozen
rerun uses the following query families and close variants:

```text
quantum tomography direct work measurement public raw data
qubit tomography microwave output power dataset
homodyne tomography direct photon energy TES same shot
Wigner tomography photon counting public data
quantum heat engine work tomography data repository
Faraday morphology quantum coherence public dataset
independent calorimetry quantum state tomography
```

The original discovery phase was conversational and iterative; a complete
search-engine result log was not retained. This file freezes a reproducible
forward search protocol, not a claim that the original ranking can be replayed
exactly.

## Inclusion rule

A candidate entered the ledger when its publication or repository plausibly
offered at least two of the following on one physical platform:

- electromagnetic/morphological response;
- quantum coherence, phase or tomography;
- direct action, energy, work, power or calorimetric measurement;
- common system/run indexing;
- a standard comparator and accessible data products.

Candidates were retained after failure so that negative acquisition results
remain inspectable.

## Eligibility rule

All six Boolean gates in `scripts/audit_public_source_eligibility.py` must pass.
The script stores the DOI or repository URL, per-gate decision and textual
reason for every candidate. Gate values are scientific classifications, not
facts downloaded from an API, and may be revised when stronger metadata or
raw records become public.

After the minimal parent-law theorem, the sixth gate no longer asks whether an
abstract Tau law has been written down. It asks whether the candidate supplies
an independent operational map from its common physical source into the typed
observables and common whitening metric required to evaluate that frozen law.
Formal-law availability therefore does not convert an experimentally
underidentified packet into an eligible score.

## Version and hash policy

DOIs and versioned repository URLs are the frozen source identifiers. External
payloads are not redistributed by Paper VII and were not hashed as a common
download corpus. Consequently this package reproduces the candidate ledger and
eligibility calculation, but not every external byte stream. A future endpoint
score must add immutable file manifests and SHA-256 hashes before analysis.

## Non-claim

The `0/35` result means that none of the frozen candidates satisfies all six
predeclared gates. It is neither an exhaustive literature statement nor a null
Tau Core result.
