# Contributing to DevWhisper

Thanks for your interest in contributing to DevWhisper! This guide covers everything you need to know to get started, whether this is your first open-source contribution or your fiftieth.

DevWhisper is a voice-native developer assistant, participating in **ECSoC '26** (Elite Coders Summer of Code). This document explains our workflow so contributions go smoothly and get reviewed quickly.

---

## Before You Start

1. **Browse open issues** — check the [Issues tab](https://github.com/Aharshi3614/Devwhisper/issues) for tasks tagged `good-first-issue`, `ECSoC26-L1`, `ECSoC26-L2`, or `ECSoC26-L3` depending on your comfort level.
2. **Wait to be assigned** — before starting work, comment on the issue you want to pick up (e.g., "I'd like to work on this") and wait for it to be assigned to you. This avoids duplicate work from multiple contributors on the same issue.
3. **Ask questions early** — if anything in an issue is unclear, comment on the issue itself rather than guessing. We're happy to clarify scope before you start coding.

---

## Setting Up Locally

1. Fork the repository and clone your fork:
   ```bash
   git clone https://github.com/<your-username>/Devwhisper.git
   cd Devwhisper
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # macOS/Linux
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and fill in required values (Qdrant URL/API key, Groq API key, etc.).
4. Run the app locally:
   ```bash
   uvicorn main:app --reload
   ```
5. Confirm it's working by visiting `/health`.

---

## Branch Naming

Use a descriptive branch name that references the issue number:
```
feature/issue-12-reset-endpoint
fix/issue-7-healthcheck-bug
docs/issue-1-readme-badges
```

---

## Making Changes

- Keep pull requests **scoped to the issue** you were assigned — avoid unrelated file changes or drive-by refactors in the same PR.
- Follow existing code style and structure (check similar files before introducing a new pattern).
- Add tests where relevant, especially for new logic in `retriever.py`, `llm.py`, or `handlers/`.
- Run linting locally before pushing:
  ```bash
  flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
  ```
- Run tests locally before pushing:
  ```bash
  pytest
  ```

---

## Pull Request Guidelines

1. **Link the issue** in your PR description using a closing keyword, e.g.:
   ```
   Resolves #12
   ```
   This auto-closes the issue when the PR merges, and also triggers our label-sync automation so your PR inherits the issue's labels (e.g., `ECSoC26-L1`).

2. **Add the `ECSoC26` label** to your PR before or at merge time — PRs without this label won't be scored by ECSoC Sentinel.

3. **Describe what changed and why** — a short summary is enough for small fixes; larger features should include what was added, how it was tested, and any tradeoffs made.

4. **Keep the diff focused** — one issue, one PR. If you find unrelated bugs while working, open a separate issue instead of fixing them in the same PR.

---

## CI & Automated Checks

Every PR runs through:
- **flake8** — catches syntax errors and undefined names (build-breaking) plus style warnings (non-blocking)
- **pytest** — runs the test suite
- **Label sync** — automatically copies labels from the linked issue onto your PR
- **ECSoC Sentinel** — scores merged PRs automatically based on difficulty label (`ECSoC26-L1` / `L2` / `L3`)

Make sure your PR passes CI (green checkmark) before requesting review. If it fails, click into the failed step's logs to see the exact error.

---

## Code Review

- A maintainer will review your PR as soon as possible.
- You may be asked to make changes — this is a normal part of the process, not a rejection. Push additional commits to the same branch and the PR updates automatically.
- Once approved, a maintainer will merge your PR.

---

## Bonus XP Labels

Maintainers may award additional XP labels on exceptional PRs:

| Label | Bonus XP |
|---|---|
| `good-issue` | +10 XP |
| `good-pr` | +15 XP |
| `good-ui` | +25 XP |
| `good-backend` | +50 XP |

These are reserved for contributions that go beyond the expected scope of the issue.

---

## Questions?

Open a discussion or comment on the relevant issue. We're happy to help — don't hesitate to ask, especially if this is your first time contributing to open source.

Happy coding! 🎙️
