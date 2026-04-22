# Contributing to Bridgekit

Bridgekit is open source and early. Contributions are welcome - whether that's a bug fix, a new tool, or an improvement to an existing one.

## Getting started

1. Fork the repo and clone it locally
2. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

3. Set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY=your_key_here
```

4. Run the tests to confirm everything is working:

```bash
pytest
```

## Making changes

- Create a new branch for your change: `git checkout -b your-branch-name`
- Keep changes focused - one fix or feature per PR
- Run `pytest` before submitting

## Project structure

```
bridgekit/
  config.py       # model and shared config
  reviewer.py     # evaluate()
  search.py       # ask()
  planner.py      # plan()
  redteam.py      # redteam()
tests/
  test_reviewer.py
  test_search.py
  test_planner.py
```

## Adding a new tool

Each tool lives in its own file under `bridgekit/`. To add one:

1. Create `bridgekit/yourtool.py` with a single public function
2. Export it from `bridgekit/__init__.py`
3. Add tests under `tests/`
4. Document it in the README with an example and sample output

## Submitting a PR

- Open a pull request against `main`
- Describe what the change does and why
- If it's a new tool, include a short example in the PR description

## Questions or ideas

Open an issue at [github.com/getbridgekit/bridgekit/issues](https://github.com/getbridgekit/bridgekit/issues) - happy to discuss before you start building.
