# hybrid-moderation: rules for agents

A comment moderator that sends each comment to a fast classifier first and asks a local LLM only about the grey
zone, measured against five single systems on Civil Comments and HateCheck. It runs on a laptop with free models.

## Check command

```bash
uv sync --frozen && uv run pytest -q
```

This is what CI runs (`.github/workflows/ci.yml`). The tests use fakes, so they need no model, no Ollama and no
Detoxify extra. Full benchmark runs need Ollama and the datasets; see the README.

- Thresholds are chosen on the tune set only. Never tune on the test sets.
- Every number in the README must come from `results/results.json`.
- This project has no AWS code. If any is added, the AWS rule below applies.

## Rules

- **Free services only.** Never call real AWS or any paid API, and never add a paid service, key or
  subscription. Models run locally (Ollama or open weights) and tests use fakes.
- **AWS goes to a local mock.** For anything AWS-backed, use the free local
  [AWS Emulator](https://github.com/aws/aws-emulator) or [Moto](https://github.com/getmoto/moto), with
  `AWS_ENDPOINT_URL` pointing at `127.0.0.1`. Never use real AWS credentials or profiles.
- **No IAM-enforcement claims.** The emulator doesn't evaluate IAM policies (and Moto doesn't by default), so a
  call that succeeds locally proves nothing about permissions. Don't write "least privilege enforced" or
  "access denied as expected" unless it's checked another way, and say so.
- **No secrets or personal data.** No keys, tokens, `.env` files, real names, emails, addresses or account
  numbers in code, data, docs or commit messages. Fixtures and demo data are fictional.
- **Report what was run.** Numbers in the README come from committed result files. If you didn't run
  something, say so instead of guessing.
- Keep changes small and reviewable, one branch and pull request per change, and don't merge your own PR.
