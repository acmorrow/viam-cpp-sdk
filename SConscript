# -*- mode: python; -*-

Import(['env'])

env = env.Clone()

env.SConscript(
    dirs=[
        'src'
    ],
    exports={
        'env' : env.Clone(),
    },
)
