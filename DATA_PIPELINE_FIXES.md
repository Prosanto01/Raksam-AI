# Data-pipeline fixes

- Redirects are rejected before an unapproved host is requested and the final
  URL is rechecked against both the domain allowlist and `robots.txt`.
- Raw HTML is saved as source-attributed JSONL in `data/web_raw/`; the dataset
  loader reads only cleaned records from `data/web_knowledge/`.
- Checkpoints record a fingerprint of their exact held-out validation examples.
  Automatic promotion is blocked unless active and candidate checkpoints have
  matching fingerprints, preventing comparisons across different validation
  sets. Existing checkpoints without fingerprints are intentionally protected.
- The conversation layer now enforces the configured brain output-token cap.
