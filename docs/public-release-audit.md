# Public Release Audit

Audit date: 2026-08-17.

This audit covers the repository refs and Git objects available in the local clone at audit time. It is a release-readiness record and does not by itself authorize changing the repository visibility.

## Findings

- Restricted row-level datasets and processed study tables: not found in the reachable repository history.
- Trained model binaries: not found in the reachable repository history. `models/production_model_manifest.json` contains metadata and local artifact paths only.
- Credentials, tokens, and private keys: not found by the automated scan of tracked files or the additional high-risk pattern scan across reachable history.
- Private infrastructure details: not found. Localhost and loopback endpoints appear only in development configuration, examples, and tests; the remaining reviewed endpoints are public documentation or repository links.
- Aggregate research outputs: present as expected. Summary tables, reports, and figures are within the allowed public artifact scope.
- Historical author metadata: multiple commits use personal mailbox addresses instead of GitHub noreply addresses. The addresses are not reproduced in this document, but they are reachable from the repository history and require an owner decision before public release.

## Checks performed

- Reviewed every ref returned by `git for-each-ref` and every commit/object reachable through `git rev-list --all`.
- Reviewed historical paths for datasets, row-level exports, serialized models, private configuration, and credential-like files.
- Ran `detect-secrets` 1.5.0 against all tracked files; the result contained no findings.
- Ran an additional automated scan for common token, private-key, and credential patterns across reachable commits and local dangling blobs; no high-risk findings were returned.
- Ran `git fsck --full --no-reflogs --unreachable`. The clone contains dangling local objects, but they are not reachable from the repository refs and the additional scan found no high-risk pattern in their blobs.
- Reviewed reachable blob sizes and paths. The largest tracked artifacts are aggregate PNG figures; no dataset or model binary was identified.

## Required action before public release

The historical personal mailbox addresses should either be explicitly accepted as intended public author metadata or removed through a coordinated history rewrite. No history rewrite was performed because it changes commit SHAs and requires repository-owner approval. Until that decision is made, this repository should remain private and this audit issue should remain open.

If a rewrite is approved, perform it from a fresh clone, update every affected ref, force-push only after coordination, and repeat this audit against the rewritten history.
