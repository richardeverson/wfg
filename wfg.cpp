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
#include <numpy/arrayobject.h>
}

#include <string>

#include "WFG/Toolkit/ExampleProblems.h"
#include "WFG/Toolkit/TransFunctions.h"
using namespace WFG::Toolkit;
using namespace WFG::Toolkit::Examples;
using std::vector;
using std::string;

/*
 * The random solution generation routines are copied directly
 * from WFG main.cpp
 */

//** Using a uniform random distribution, generate a number in [0,bound]. ***

double next_double( const double bound = 1.0 )
{
  assert( bound > 0.0 );

  return bound * rand() / static_cast< double >( RAND_MAX );
}


//** Create a random Pareto optimal solution for WFG1. **********************

vector< double > WFG_1_random_soln( const int k, const int l )
{
  vector< double > result;  // the result vector


  //---- Generate a random set of position parameters.

  for( int i = 0; i < k; i++ )
    {
      // Account for polynomial bias.
      result.push_back( pow( next_double(), 50.0 ) );
    }


  //---- Set the distance parameters.

  for( int i = k; i < k+l; i++ )
    {
      result.push_back( 0.35 );
    }


  //---- Scale to the correct domains.

  for( int i = 0; i < k+l; i++ )
    {
      result[i] *= 2.0*(i+1);
    }


  //---- Done.

  return result;
}


//** Create a random Pareto optimal solution for WFG2-WFG7. *****************

vector< double > WFG_2_thru_7_random_soln( const int k, const int l )
{
  vector< double > result;  // the result vector


  //---- Generate a random set of position parameters.

  for( int i = 0; i < k; i++ )
    {
      result.push_back( next_double() );
    }


  //---- Set the distance parameters.

  for( int i = k; i < k+l; i++ )
    {
      result.push_back( 0.35 );
    }


  //---- Scale to the correct domains.

  for( int i = 0; i < k+l; i++ )
    {
      result[i] *= 2.0*(i+1);
    }


  //---- Done.

  return result;
}


//** Create a random Pareto optimal solution for WFG8. **********************

vector< double > WFG_8_random_soln( const int k, const int l )
{
  vector< double > result;  // the result vector


  //---- Generate a random set of position parameters.

  for( int i = 0; i < k; i++ )
    {
      result.push_back( next_double() );
    }


  //---- Calculate the distance parameters.

  for( int i = k; i < k+l; i++ )
    {
      const vector< double >  w( result.size(), 1.0 );
      const double u = TransFunctions::r_sum( result, w  );

      const double tmp1 = fabs( floor( 0.5 - u ) + 0.98/49.98 );
      const double tmp2 = 0.02 + 49.98*( 0.98/49.98 - ( 1.0 - 2.0*u )*tmp1 );

      result.push_back( pow( 0.35, pow( tmp2, -1.0 ) ));
    }


  //---- Scale to the correct domains.

  for( int i = 0; i < k+l; i++ )
    {
      result[i] *= 2.0*(i+1);
    }


  //---- Done.

  return result;
}


//** Create a random Pareto optimal solution for WFG9. **********************

vector< double > WFG_9_random_soln( const int k, const int l )
{
  vector< double > result( k+l );  // the result vector


  //---- Generate a random set of position parameters.

  for( int i = 0; i < k; i++ )
    {
      result[i] = next_double();
    }


  //---- Calculate the distance parameters.

  result[k+l-1] = 0.35;  // the last distance parameter is easy

  for( int i = k+l-2; i >= k; i-- )
    {
      vector< double > result_sub;
      for( int j = i+1; j < k+l; j++ )
        {
          result_sub.push_back( result[j] );
        }

      const vector< double > w( result_sub.size(), 1.0 );
      const double tmp1 = TransFunctions::r_sum( result_sub, w  );

      result[i] = pow( 0.35, pow( 0.02 + 1.96*tmp1, -1.0 ) );
    }


  //---- Scale to the correct domains.

  for( int i = 0; i < k+l; i++ )
    {
      result[i] *= 2.0*(i+1);
    }


  //---- Done.

  return result;
}


//** Create a random Pareto optimal solution for I1. *****************

vector< double > I1_random_soln( const int k, const int l )
{
  vector< double > result;  // the result vector


  //---- Generate a random set of position parameters.

  for( int i = 0; i < k; i++ )
    {
      result.push_back( next_double() );
    }


  //---- Set the distance parameters.

  for( int i = k; i < k+l; i++ )
    {
      result.push_back( 0.35 );
    }


  //---- Done.

  return result;
}


//** Create a random Pareto optimal solution for I2. **********************

vector< double > I2_random_soln( const int k, const int l )
{
  vector< double > result( k+l );  // the result vector


  //---- Generate a random set of position parameters.

  for( int i = 0; i < k; i++ )
    {
      result[i] = next_double();
    }


  //---- Calculate the distance parameters.

  result[k+l-1] = 0.35;  // the last distance parameter is easy

  for( int i = k+l-2; i >= k; i-- )
    {
      vector< double > result_sub;
      for( int j = i+1; j < k+l; j++ )
        {
          result_sub.push_back( result[j] );
        }

      const vector< double > w( result_sub.size(), 1.0 );
      const double tmp1 = TransFunctions::r_sum( result_sub, w  );

      result[i] = pow( 0.35, pow( 0.02 + 1.96*tmp1, -1.0 ) );
    }


  //---- Done.

  return result;
}


//** Create a random Pareto optimal solution for I3. **********************

vector< double > I3_random_soln( const int k, const int l )
{
  vector< double > result;  // the result vector


  //---- Generate a random set of position parameters.

  for( int i = 0; i < k; i++ )
    {
      result.push_back( next_double() );
    }


  //---- Calculate the distance parameters.

  for( int i = k; i < k+l; i++ )
    {
      const vector< double >  w( result.size(), 1.0 );
      const double u = TransFunctions::r_sum( result, w  );

      const double tmp1 = fabs( floor( 0.5 - u ) + 0.98/49.98 );
      const double tmp2 = 0.02 + 49.98*( 0.98/49.98 - ( 1.0 - 2.0*u )*tmp1 );

      result.push_back( pow( 0.35, pow( tmp2, -1.0 ) ));
    }


  //---- Done.

  return result;
}


//** Create a random Pareto optimal solution for I4. **********************

vector< double > I4_random_soln( const int k, const int l )
{
  return I1_random_soln( k, l );
}


//** Create a random Pareto optimal solution for I5. **********************

vector< double > I5_random_soln( const int k, const int l )
{
  return I3_random_soln( k, l );
}

//** Generate a random solution for a given problem. ************************

vector< double > problem_random_soln
(
 const int k,
 const int l,
 const std::string fn
 )
{
  if ( fn == "WFG1" )
    {
      return WFG_1_random_soln( k, l );
    }
  else if
    (
     fn == "WFG2" ||
     fn == "WFG3" ||
     fn == "WFG4" ||
     fn == "WFG5" ||
     fn == "WFG6" ||
     fn == "WFG7"
     )
    {
      return WFG_2_thru_7_random_soln( k, l );
    }
  else if ( fn == "WFG8" )
    {
      return WFG_8_random_soln( k, l );
    }
  else if ( fn == "WFG9" )
    {
      return WFG_9_random_soln( k, l );
    }
  else if ( fn == "I1" )
    {
      return I1_random_soln( k, l );
    }
  else if ( fn == "I2" )
    {
      return I2_random_soln( k, l );
    }
  else if ( fn == "I3" )
    {
      return I3_random_soln( k, l );
    }
  else if ( fn == "I4" )
    {
      return I4_random_soln( k, l );
    }
  else if ( fn == "I5" )
    {
      return I5_random_soln( k, l );
    }
  else
    {
      assert( false );
      return vector< double >();
    }
}


/*
 *  Wrapping functions here.
 */



static PyObject*
wfg_random_soln(PyObject *self, PyObject *args, PyObject *kwds)
{
  int k, l, N;
  char *problem;

  if(!PyArg_ParseTuple(args, "iis", &k, &l, &problem)) {
    return NULL;
  }

  /* Get a random solution of length k+l */
  vector<double> zvector = problem_random_soln(k, l, problem);
  N = zvector.size();           /* Should be k+l */
  /* New numpy array for the result */
  npy_intp dims[1] = {N};
  PyArrayObject *zarray =
    (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
  double *z = (double *)zarray->data;
  for (int j = 0; j < N; j++)
    z[j] = zvector[j];

  return PyArray_Return(zarray);
}                           




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
    "Walking Fish Group test problem " QUOTE(name) "\n\n"                 \
    "Signature:  y = " QUOTE(name) "(z, k, M)\n"                         \
    "\n"                                                                \
    "z        decision vector as a Numpy array\n"                       \
    "k        number of position parameters\n"                          \
    "M        number of objectives\n"                                   \
    "\n"                                                                \
    "y        Numpy array of M objective evaluations\n"                 \
    "\n"                                                                \
    "Parameters must satisfy:  0 < k < N and M >= 2 and k % (M-1) == 0\n" \
    "where N = len(z)"                                                \
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

  {"random_soln", (PyCFunction)wfg_random_soln,
   METH_VARARGS|METH_KEYWORDS,
   "Random Pareto optimal solutions for WFG problems\n"
   "Signature:   z = random_soln(k, l, problem)\n"
   "\n"
   "k        number of position parameters\n"
   "l        number of distance parameters\n"
   "problem  string giving the problem name, eg 'WFG2' or 'I3'\n"
   "\n"
   "z        decision variable vector of length k+l"
  },

  {NULL}  /* Sentinel */
};


#define MODULE_DOC \
"Walking Fish Group test problems\n"                            \
"\n"                                                            \
"A simple wrapper around the WFG test problems\n"               \
"Objective evaluation for problems WFG1 to WFG9 and I1 to I3\n"   \
"is provided by the functions WFG1(), etc\n"                    \
"Random Pareto optimal solutions can be generated\n"            \
"with the function 'random_soln()'\n\n"                         \

#if PY_MAJOR_VERSION >= 3
  static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "wfg",                      /* m_name */
    MODULE_DOC,                 /* m_doc */
    -1,                         /* m_size */
    wfg_methods,                /* m_methods */
    NULL,                       /* m_reload */
    NULL,                       /* m_traverse */
    NULL,                       /* m_clear */
    NULL,                       /* m_free */
  };
#endif

PyMODINIT_FUNC
PyInit_wfg(void) 
{
  PyObject* m;
  /* Initialize the module */
  m = PyModule_Create(&moduledef);
  if (m == NULL) return NULL;

  import_array();
  return m;
}

}
