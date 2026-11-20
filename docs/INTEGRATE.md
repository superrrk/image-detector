# Integrate with the original GitHub repository

The source repository could be read, but its GitHub connection denied branch creation. No repository commits were made.

To adopt this complete replacement in a fresh branch without overwriting existing research files:

1. Clone `https://github.com/superrrk/image-detector.git`.
2. `git checkout -b feature/image-forensics-lab`.
3. Copy everything in this ZIP's `image-forensics-lab/` directory into the repository root. Allow its `server/app.py`, `README.md`, `.gitignore`, and `requirements.txt` to replace the old versions.
4. Leave `predict.py`, `train*.py`, `prepare_data.py`, and existing dataset files untouched for historical research purposes. The old React `client` submodule is not used by the new single-server UI.
5. Follow the README Quick Start instructions, then run `python scripts/smoke_test.py` and `python -m pytest -q` after installing the dependencies.
6. Commit and open a pull request against `main`.

Do not rewrite large Git history without a separate review. The committed dataset makes the original repository very large; consider migrating it to external storage after verifying licensing and provenance.
