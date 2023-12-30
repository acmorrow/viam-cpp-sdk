# -*- mode: python; -*-

import SCons.Tool


def generate(env) -> None:
    """Add post-default tools."""
    for t in env["POSTDEFAULT_TOOLS"]:
        SCons.Tool.Tool(t)(env)


def exists(env) -> bool:
    return True
