# Releasing Cascade

The steps that make a release public. Everything before them (tests, build and package checks, a
clean-install test) is done for 0.12.0. Run the commands from the repository root in PowerShell.

## 0. Personal data in the history (done for 0.12.0)

Before the repository went public (2026-11-27), the history was scanned. No secrets (API keys,
tokens, private keys) were found. Two personal email addresses were removed by rewriting the history
with `git filter-repo`: commit author emails were mapped to GitHub noreply addresses, and an address
inside an archived Word document was replaced. Before each future release, scan new commits the same
way.

## 1. Check

```powershell
python -m pytest -q
```

```powershell
python -m build
```

```powershell
python -m twine check dist/*
```

## 2. Make the repository public (optional, owners only)

```powershell
gh repo edit SMaarij/Exocuda --visibility public --accept-visibility-change-consequences
```

## 3. Upload to PyPI

Create an account at https://pypi.org, then make an API token under Account settings → API tokens.
Use the token as the password, with `__token__` as the username.

```powershell
python -m twine upload dist/*
```

To rehearse first, upload to TestPyPI (https://test.pypi.org) and install from there:

```powershell
python -m twine upload --repository testpypi dist/*
```

```powershell
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple cascade-local
```

## 4. Tag the release

```powershell
git tag v0.12.0
```

```powershell
git push origin v0.12.0
```

```powershell
gh release create v0.12.0 dist/* --title "Cascade 0.12.0" --notes-file CHANGELOG.md
```

## Next releases

Bump `version` in `pyproject.toml` and `__version__` in `cascade_local/__init__.py`, add a
`CHANGELOG.md` entry, delete the old `dist/`, and repeat steps 1, 3 and 4.
