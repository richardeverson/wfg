/* WFG -- the Walking Fish Group multi-objective test problems 
 *
 *  Copyright (C) 2010 Richard Everson
 *  All rights reserved.
 *
 *  Richard Everson <R.M.Everson@exeter.ac.uk>
 *  College of Engineering, Mathematics and Physical Sciences,
 *  University of Exeter,  Exeter, UK. EX4 4QF
 *
 *  NOTICE
 *
 *  This library is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU Library General Public
 *  License as published by the Free Software Foundation; either
 *  version 2 of the License, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *  Library General Public License for more details.
 *
 *  You should have received a copy of the GNU Library General Public
 *  License along with this library; if not, write to the
 *  Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 *   Boston, MA 02111-1307, USA.
 */

extern "C" {
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <Python.h>
#include <structmember.h>
#include "numpy/arrayobject.h"
}

#include "WFG/Toolkit/ExampleProblems.h"
#include "WFG/Toolkit/TransFunctions.h"
using namespace WFG::Toolkit;
using namespace WFG::Toolkit::Examples;
using std::vector;




#define QUOTE(s) # s   /* turn s into string "s" */

/* Directives to join symbols -- see https://www.securecoding.cert.org/confluence/display/seccode/PRE05-C.+Understand+macro+replacement+when+concatenating+tokens+or+performing+stringification */

#define JOIN(x, y) JOIN_AGAIN(x, y)
#define JOIN_AGAIN(x, y)  x ## y
/* Macro to define the interface for a single problem */ 
#define WFGproblem(name)                                            \
                                                                    \
static PyObject*                                                    \
 JOIN(wfg_, name)                                                   \
(PyObject *self, PyObject *args, PyObject *kwds)                    \
{                                                                   \
  PyObject *Oz = NULL;                                              \
  PyArrayObject *zarray = NULL;                                     \
  int k, M;                                                         \
                                                                    \
  if(!PyArg_ParseTuple(args, "Oii", &Oz, &k, &M)) {                 \
    return NULL;                                                    \
  }                                                                 \
                                                                    \
  if(!(zarray = (PyArrayObject *)                                   \
       PyArray_ContiguousFromObject(Oz, PyArray_DOUBLE, 1, 1))) {   \
    return NULL;                                                    \
  }                                                                 \
                                                                    \
  int N = zarray->dimensions[0];                                    \
  double *z = (double *)zarray->data;                               \
  vector<double> zvector(N);                                        \
  zvector.assign(z, z+N);                                           \
                                                                    \
  /* Call it */                                                     \
  vector<double> yvector = Problems::name( zvector, k, M);          \
                                                                    \
  /* New numpy array for the result */                              \
  npy_intp dims[1] = {M};                                           \
  PyArrayObject *yarray =                                           \
    (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);        \
  double *y = (double *)yarray->data;                               \
  for (int j = 0; j < M; j++)                                       \
    y[j] = yvector[j];                                              \
                                                                    \
  return PyArray_Return(yarray);                                    \
}                           

WFGproblem(WFG1)
WFGproblem(WFG2)
WFGproblem(WFG3)
WFGproblem(WFG4)
WFGproblem(WFG5)
WFGproblem(WFG6)
WFGproblem(WFG7)
WFGproblem(WFG8)
WFGproblem(WFG9)
WFGproblem(I1)
WFGproblem(I2)
WFGproblem(I3)
WFGproblem(I4)


/* Preprocessor macro for PyMethodDef */

#define METHOD_DEF(name)                                                \
{QUOTE(name),  (PyCFunction)JOIN(wfg_, name),                           \
    METH_VARARGS|METH_KEYWORDS,                                         \
    "Walking Fish Group test problem " QUOTE(name) "\n"                 \
    "Signature: y = " QUOTE(name) "(z, k, M)\n"                         \
    "\n"                                                                \
    "z        decision vector as a Numpy array\n"                       \
    "k        number of position parameters\n"                          \
    "M        number of objectives\n"                                   \
    "\n"                                                                \
    "y        Numpy array of M objective evaluations\n"                 \
    "\n"                                                                \
    "Parameters must satisfy:  0 < k < N and M >= 2 and k % (M-1) == 0\n" \
    "where N = len(z)\n"                                                \
    }


extern "C" {

static PyMethodDef wfg_methods[] = {

  METHOD_DEF(WFG1),
  METHOD_DEF(WFG2),
  METHOD_DEF(WFG3),
  METHOD_DEF(WFG4),
  METHOD_DEF(WFG5),
  METHOD_DEF(WFG6),
  METHOD_DEF(WFG7),
  METHOD_DEF(WFG8),
  METHOD_DEF(WFG9),
  METHOD_DEF(I1),
  METHOD_DEF(I2),
  METHOD_DEF(I3),
  METHOD_DEF(I4),

  {NULL}  /* Sentinel */
};

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
init_wfg(void) 
{
  PyObject* m;

  /* Initialize the module */
  m = Py_InitModule3("_wfg", wfg_methods,"WFG test functions");
  if (m == NULL) return;

  import_array()

}

}
