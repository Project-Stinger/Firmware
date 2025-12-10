"""
Minimal gitVersion script for PlatformIO extra_scripts.
This placeholder avoids the missing SConscript error. It attempts to read a git description
and attach it as a preprocessor define if possible, but otherwise silently does nothing.
"""
import subprocess

def _get_git_version():
    try:
        return subprocess.check_output(["git", "describe", "--tags", "--always"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None

git_ver = _get_git_version()
if git_ver:
    try:
        # If PlatformIO provides an SCons env, append a define. This is optional.
        env = globals().get("env")
        if env is not None:
            env.Append(CPPDEFINES=[("GIT_VERSION", '"%s"' % git_ver)])
    except Exception:
        pass
