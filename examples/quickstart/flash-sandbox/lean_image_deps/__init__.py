# Empty. Exists only so `uv pip install '.[verl,harbor]'` (setuptools backend)
# has a package to build -- the real deps come from [project.optional-dependencies].
# Never imported at runtime (verl/uni_agent are on the mounted repo via PYTHONPATH).
