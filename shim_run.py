# shim_run.py
import os, runpy
import MetaTrader5 as mt5

# Keep the real function to fall back on
_real_initialize = mt5.initialize

def _patched_initialize(*args, **kwargs):
    """
    If the following env vars are set, we override the script's MT5.initialize args:
      MT5_TERMINAL_PATH, MT5_LOGIN, MT5_PASSWORD, MT5_SERVER
    If they're not set, we do nothing and let the script use its own values.
    """
    term_path = os.environ.get("MT5_TERMINAL_PATH")
    login     = os.environ.get("MT5_LOGIN")
    password  = os.environ.get("MT5_PASSWORD")
    server    = os.environ.get("MT5_SERVER")

    if term_path: kwargs["path"] = term_path
    if login:     kwargs["login"] = int(login)
    if password:  kwargs["password"] = password
    if server:    kwargs["server"] = server

    return _real_initialize(*args, **kwargs)

# Monkey-patch
mt5.initialize = _patched_initialize

# Run the target script specified by the launcher
target = os.environ["TARGET_SCRIPT"]
runpy.run_path(target, run_name="__main__")