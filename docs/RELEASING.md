# Releasing Cascade

The steps that make a release public. Everything before them (tests, build and package checks, a
clean-install test) is done for 0.12.0. Run the commands from the repository root in PowerShell.

## 0. Before going public: personal data in the history

Making the repository public also publishes its whole git history. A scan found no secrets (API
keys, tokens, private keys) in any commit. It did find two personal email addresses:

- Jaweria's Gmail address is the author email on 2 commits.
- An early commit (`6325b4f`) had a Gmail address in a file. It was removed later but is still in
  the history.

Either accept that, or rewrite the history before going public (for example with
`git filter-repo --mailmap`). A rewrite changes every commit ID and needs a force-push, so agree it
with everyone who has a clone. No rewrite has been done.

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
