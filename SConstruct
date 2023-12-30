# -*- mode: python; -*-

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

# --variants-paths=p1,p2...pn
# --variants=a,b,c
# 

#
# SCons options are set here.
#

# Always randomize build order, to shake out dependency issues and to
# avoid racing on the cache when two builders are building
# concurrently but sharing a cache.
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
    'CACHE_DIR',
    default='$SCONS_DIR/cache',
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
    converter=shlex.split,
)

variables.Add(
    'DEST_DIR',
    default='$BUILD_DIR/install',
)

variables.Add(
    ('PLUS_CPPPATH', '+CPPPATH'),
    default=[],
    converter=shlex.split,
)

variables.Add(
    ('CPPPATH_PLUS', 'CPPPATH+'),
    default=[],
    converter=shlex.split,
)

variables.Add(
    'LIBPATH',
    default=None,
    converter=shlex.split,
)

variables.Add(
    'PLATFORM',
    default=None,
)

variables.Add(
    'PREFIX',
    default="/"
)

variables.Add(
    'PREFIX_BIN_DIR',
    default='$PREFIX/bin',
)

variables.Add(
    'PREFIX_INCLUDE_DIR',
    default='$PREFIX/include',
)

variables.Add(
    'PREFIX_LIB_DIR',
    default='$PREFIX/lib'
)

variables.Add(
    'SCONS_DIR',
    default='$BUILD_DIR/scons',
)

variables.Add(
    'TOOLS',
    converter=lambda x: ['predefault'] + shlex.split(x) + ['postdefault'],
    default=['default']
)

variables.Add(
    ('PREDEFAULT_TOOLS', '+TOOLS'),
    converter=shlex.split,
    default=[],
)

variables.Add(
    ('POSTDEFAULT_TOOLS', 'TOOLS+'),
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
    VARIANT_DIR='$BUILD_DIR/variants/$VARIANT',
)

# Apply any PLUS_ or _PLUS variables, per our convention
fixup = env.Clone()
for k,v in env.items():
    if k.startswith('PLUS_'):
        fixup.Prepend(**{k[len('PLUS_'):] : v})
        del fixup[k]
    elif k.endswith('_PLUS'):
        fixup.Append(**{k[:len('_PLUS')] : v})
        del fixup[k]
env = fixup


#
# SCons configuration goes here
#

env.SConsignFile('${__env__.Dir(SCONS_DIR)}/sconsign')
env.CacheDir('${__env__.Dir(CACHE_DIR)}')


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
