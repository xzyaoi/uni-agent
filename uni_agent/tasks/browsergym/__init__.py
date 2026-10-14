"""BrowserGym task family: train agents on web tasks with a real browser in a sandbox.

One episode = one :class:`~browsergym.core.env.BrowserEnv` task running inside a
sandbox (a headless Chromium + the bundled MiniWoB++ sites, or a client to
externally hosted sites like WebArena). The agent acts through the ``browser``
tool (one BrowserGym action string per call); reward is the env's own episode
reward (sparse 0/1 success for MiniWoB/WebArena).
"""
