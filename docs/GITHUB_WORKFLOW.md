# GitHub workflow

GitHub Desktop records the project history; it does not perform the statistics.

1. Open this project folder in VS Code and run the replication and tests.
2. Open the same folder in GitHub Desktop.
3. Review the changed files and the generated `results/` tables and figures.
4. Use the summary `Initial PCA-HMM replication pipeline` and commit to `main`.
5. Choose **Publish repository** once, then **Push origin** for later commits.

Each commit should represent a meaningful checkpoint, for example:

- `Add PCA factor construction and variance audit`
- `Add Gaussian HMM training and state forecast`
- `Add rolling backtest, tests, and paper audit`

Do not commit credentials, private market-data tokens, or large raw price
downloads. Keep the data provenance and date range in a small text file or
README instead.
