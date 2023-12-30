# -*- mode: python; -*-

import SCons.Tool


def generate(env) -> None:
    if "BREW" not in env:
        env["BREW"] = env.WhereIs("brew")
    env.AppendUnique(CPPPATH=["/opt/homebrew/include"])
    env.AppendUnique(LIBPATH=["/opt/homebrew/lib"])


def exists(env) -> bool:
    if "BREW" in env and env["BREW "] is not None:
        return True
    return env.WhereIs("brew") != None
