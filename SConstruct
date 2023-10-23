import shlex

# Bypass expensive tool initialization on startup
DefaultEnvironment(tools=[])

# Configure minimum python and SCons versions
EnsurePythonVersion(3, 8)
EnsureSConsVersion(4, 5)

#
# Global options declarations begin here. Please keep declared options in alphabetical order. Options
# local to a sub-tree should be declared in the associated SConstruct.
#

#
# SCons options are set here.
#

SetOption('random', 1)



#
# Global variable declarations begin here. Please keep variables in alphabetical order. Variables
# local to a sub-tree should be declared in the associated SConstruct.
#

# TODO: Variable Files!
variables = Variables(
    args=ARGUMENTS
)

variables.Add(
    'BUF',
    default=WhereIs('buf'),
)

variables.Add(
    'BUILD_DIR',
    default='#/build',
)

variables.Add(
    'CCFLAGS',
    converter=shlex.split,
)

variables.Add(
    'CFLAGS',
    converter=shlex.split,
)

variables.Add(
    'CXXFLAGS',
    converter=shlex.split,
)

variables.Add(
    'CPPPATH',
    default=[],
    converter=shlex.split,
)

variables.Add(
    'LIBPATH',
    default=[],
    converter=shlex.split,
)

variables.Add(
    'TOOLS',
    converter=shlex.split,
    default=[],
)

variables.Add(
    'VARIANT',
    default=None,
)


#
# Environment construction begins here
#

env = Environment(
    variables=variables,
    VARIANT_DIR='$BUILD_DIR/$VARIANT',
)


#
# Jump to SConscript so our root is in the variant directory
#

env.SConscript(
    dirs=[
        '.',
    ],
    duplicate=False,
    exports={
        'env' : env.Clone(),
    },
    variant_dir='$VARIANT_DIR',
)
