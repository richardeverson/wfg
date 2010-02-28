#! /usr/bin/env python

"""WFG -- the Walking Fish Group multi-objective test problems 

  Copyright (C) 2010 Richard Everson
  All rights reserved.

  Richard Everson <R.M.Everson@exeter.ac.uk>
  College of Engineering, Mathematics and Physical Sciences,
  University of Exeter,  Exeter, UK. EX4 4QF

NOTICE

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Library General Public
  License as published by the Free Software Foundation; either
  version 2 of the License, or (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Library General Public License for more details.

  You should have received a copy of the GNU Library General Public
  License along with this library; if not, write to the
  Free Software Foundation, Inc., 59 Temple Place - Suite 330,
  Boston, MA 02111-1307, USA.
"""


from distutils.core import setup, Extension
from distutils import sysconfig
import commands
import os, sys
import numpy, numpy.distutils
import warnings
from distutils.sysconfig import get_config_var, get_config_vars


try:
    os.system("bzr version-info --format python > _version.py")
except:
    warn.warning("Could not run bzr to get most recent version info")

import _version

VERSION =  _version.version_info['revision_id']

print "Version is ", VERSION
PYGTS_HAS_NUMPY = '0'  # Numpy detected below

# Hand-code these lists if the auto-detection below doesn't work
INCLUDE_DIRS = []
LIB_DIRS = []
LIBS = []

# numpy stuff
numpy_include_dirs = numpy.distutils.misc_util.get_numpy_include_dirs()
flag = False
for path in numpy_include_dirs:
    if os.path.exists(os.path.join(path,'numpy/arrayobject.h')):
        PYGTS_HAS_NUMPY = '1'
        break

if PYGTS_HAS_NUMPY == '1':
    INCLUDE_DIRS.extend(numpy_include_dirs)
else:
    raise RuntimeError, 'Numpy not found'


# Test for Python.h
python_inc_dir = sysconfig.get_python_inc()
if not python_inc_dir or \
        not os.path.exists(os.path.join(python_inc_dir, "Python.h")):
    raise RuntimeError, 'Python.h not found'

if sys.platform.startswith("darwin"):
    get_config_vars()['CFLAGS'] = get_config_vars()['CFLAGS'].replace('-Wno-long-double', '')

get_config_vars()['CFLAGS'] = get_config_vars()['CFLAGS'].replace('-Wstrict-prototypes', '')

# Run the setup
setup(name='wfg', 
      version=VERSION,
      description="WFG wraps the Walking Fish Group multi-objective test problems",
      long_description="WFG wraps the Walking Fish Group multi-objective test problems",
      author='Richard Everson',
      author_email='R.M.Everson@exeter.ac.uk',
      license='GNU Library General Public License (LGPL) version 2 or higher',
      url=None,
      download_url=None,
      platforms='Platform-Independent',
      py_modules=[],
      classifiers = ['Development Status :: 3 - Alpha',
                     'Intended Audience :: Developers',
                     'Intended Audience :: Science/Research',
                     'License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)',
                     'Operating System :: OS Independent',
                     'Programming Language :: C++',
                     'Programming Language :: Python',
                     'Topic :: Scientific/Engineering :: Mathematics',
                     'Topic :: Scientific/Engineering :: Optimization',
                     ],
      ext_modules=[Extension("wfg", ["wfg.cpp",
                                      "WFG/Toolkit/ExampleProblems.cpp",
                                      "WFG/Toolkit/ExampleShapes.cpp",
                                      "WFG/Toolkit/ExampleTransitions.cpp",
                                      "WFG/Toolkit/FrameworkFunctions.cpp",
                                      "WFG/Toolkit/Misc.cpp",
                                      "WFG/Toolkit/ShapeFunctions.cpp",
                                      "WFG/Toolkit/TransFunctions.cpp"
                                      ],
                             define_macros=[ ('WFG_VERSION',
                                              VERSION) ],
                             include_dirs = INCLUDE_DIRS,
                             library_dirs = LIB_DIRS,
                             libraries=LIBS)
                   ]
      )
