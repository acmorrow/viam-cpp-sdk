# -*- mode: python; -*-

import SCons.Tool


def generate(env) -> None:
    """Add pre-default tools."""
    for t in env["PREDEFAULT_TOOLS"]:
        SCons.Tool.Tool(t)(env)


def exists(env) -> bool:
    return True
