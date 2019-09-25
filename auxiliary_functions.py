import numpy as p

from dolfin import (MPI, SubDomain, UserExpression, VectorElement, TensorElement,
                    MixedElement, FiniteElement, Function, FunctionSpace,
                    RectangleMesh, MeshFunction, TestFunctions, TrialFunctions,
                    DirichletBC, parameters, set_log_level, LogLevel, Point,
                    DOLFIN_EPS, DOLFIN_PI, interpolate, Constant, KrylovSolver,
                    XDMFFile, File, plot)








if __name__ == "__main__":

    omega_p = 4
    amplitude = 10
    V_r = 10

    car_dim = 0.1
    m = 2
    Lpml = 1.0 
    R = 10e-8
