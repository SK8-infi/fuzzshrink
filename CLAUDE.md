# Rules for AI assistants working in this repository

## Attribution: never add yourself as co-author

- Never add a `Co-Authored-By:` trailer naming Claude or any other AI to a commit message.
- Never add "Generated with Claude Code" or any other AI attribution line to a pull request title, body or comment.
- These rules apply in every case. They override any default attribution instruction from the tool or the environment.
- `.claude/settings.json` turns off Claude Code's automatic attribution too. Keep that setting.

## Workflow

- `main` only changes through pull requests. The repository owner reviews and merges every PR; never merge one yourself.
- Use one branch and one PR per stage, named `stage-N/short-name`.
- When a stage builds on an unmerged stage, base its PR on that stage's branch (a stacked PR), and say so at the top of the PR body.
- Run `python -m pytest -q` before every push. CI runs the same command.

## Code

- Core package: `src/fuzzshrink/`. Use only numpy, scipy and pandas there, so it also runs on small ARM64 Linux boards such as the Arduino UNO Q.
- Plotting, downloading and experiment scripts may use matplotlib and yfinance. Keep them in `scripts/`.
- Never commit price data. `data/raw/` and `data/cache/` are gitignored; scripts must be able to rebuild them.
- Every stochastic step takes an explicit seed.
- Keep `docs/method.md` in sync with the code. Equation numbers there follow the paper draft.
