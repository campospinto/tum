import time
import numpy as np
from sympy  import pi, sin, cos, Tuple

from sympde.calculus      import dot, curl, cross
from sympde.topology      import VectorFunctionSpace
from sympde.topology      import elements_of
from sympde.topology      import NormalVector
from sympde.topology      import Square
from sympde.expr.expr     import LinearForm, BilinearForm
from sympde.expr.expr     import integral
from sympde.expr.equation import find

from psydac.api.discretization       import discretize
from psydac.api.settings             import PSYDAC_BACKEND_GPYCCEL
from psydac.linalg.solvers           import inverse

#==============================================================================
def run_maxwell_2d(uex, f, alpha, domain, ncells, degree, kappa):

    #+++++++++++++++++++++++++++++++
    # 1. Abstract model
    #+++++++++++++++++++++++++++++++

    V  = VectorFunctionSpace('V', domain, kind='hcurl')

    u, v    = elements_of(V, names='u, v')
    nn      = NormalVector('nn')

    m = BilinearForm((u, v), integral(domain, dot(u, v)))
    k = LinearForm(v, integral(domain, dot(uex, v)))

    boundary = domain.boundary

    expr1   = curl(u) * curl(v)
    expr1_b = kappa * cross(nn, u) * cross(nn, v)

    expr2   = dot(f, v)

    # Bilinear form a: V x V --> R
    a = BilinearForm((u, v),  integral(domain, expr1) + integral(boundary, expr1_b))
    
    # Linear form l: V --> R
    l = LinearForm(v, integral(domain, expr2))

    equation = find(u, forall=v, lhs=a(u, v), rhs=l(v))

    #+++++++++++++++++++++++++++++++
    # 2. Discretization
    #+++++++++++++++++++++++++++++++

    domain_h = discretize(domain, ncells=ncells)
    Vh       = discretize(V, domain_h, degree=degree, basis='M')

    equation_h = discretize(equation, domain_h, [Vh, Vh], backend=PSYDAC_BACKEND_GPYCCEL)
    m_h        = discretize(m, domain_h, [Vh, Vh], backend=PSYDAC_BACKEND_GPYCCEL)
    M          = m_h.assemble()
    k_h        = discretize(k, domain_h, Vh, backend=PSYDAC_BACKEND_GPYCCEL)
    K          = k_h.assemble()
    M_inv      = inverse(M, 'cg', tol=1e-8)
    uex_coeffs = M_inv @ K

    equation_h.assemble()
    jacobi_pc = equation_h.linear_system.lhs.diagonal(inverse=True)
    equation_h.set_solver('pcg', pc=jacobi_pc, tol=1e-8, info=True)

    timing   = {}
    t0       = time.time()
    uh, info = equation_h.solve()
    t1       = time.time()
    timing['solution'] = t1-t0

    t0 = time.time()
    l2_error = np.sqrt((uex_coeffs - uh.coeffs).dot(M @ (uex_coeffs - uh.coeffs)))
    t1       = time.time()
    timing['diagnostics'] = t1-t0

    return uh, info, timing, l2_error

#==============================================================================
if __name__ == '__main__':

    from sympy              import lambdify
    from sheet1_utils       import plot

    domain = Square('S', bounds1=(0, 1), bounds2=(0, 1))
    x,y    = domain.coordinates
    omega  = 1.5
    alpha  = -omega**2
    Eex    = Tuple(sin(pi*y), sin(pi*x)*cos(pi*y))
    f      = Tuple(alpha*sin(pi*y) - pi**2*sin(pi*y)*cos(pi*x) + pi**2*sin(pi*y),
                  alpha*sin(pi*x)*cos(pi*y) + pi**2*sin(pi*x)*cos(pi*y))

    ne     = [8, 8] # number of Bspline cells
    degree = [2, 2] # Bspline degree
    kappa  = 1e6    # penalization parameter

    Eh, info, timing, l2_error = run_maxwell_2d(Eex, f, alpha, domain, ncells=ne, degree=degree, kappa=kappa)

    # ...
    print( '> Grid          :: [{ne1},{ne2}]'.format( ne1=ne[0], ne2=ne[1]) )
    print( '> Degree        :: [{p1},{p2}]'  .format( p1=degree[0], p2=degree[1] ) )
    print( '> CG info       :: ',info )
    print( '> L2 error      :: {:.2e}'.format( l2_error ) )
    print( '' )
    print( '> Solution time :: {:.2e}'.format( timing['solution'] ) )
    print( '> Evaluat. time :: {:.2e}'.format( timing['diagnostics'] ) )
        
    Eex_x   = lambdify(domain.coordinates, Eex[0])
    Eex_y   = lambdify(domain.coordinates, Eex[1])
    Eh_x    = Eh[0]
    Eh_y    = Eh[1]
    error_x = lambda x, y: abs(Eex_x(x, y) - Eh_x(x, y))
    error_y = lambda x, y: abs(Eex_y(x, y) - Eh_y(x, y))

    plot(gridsize_x     = 100, 
          gridsize_y    = 100, 
          title         = r'approximation of solution $u$, $x$ component', 
          funs          = [Eex_x, Eh_x, error_x], 
          titles        = [r'$u^{ex}_x(x,y)$', r'$u^h_x(x,y)$', r'$|(u^{ex}-u^h)_x(x,y)|$'],
          surface_plot  = True
    )
    
    plot(gridsize_x     = 100,
          gridsize_y    = 100,
          title         = r'approximation of solution $u$, $y$ component',
          funs          = [Eex_y, Eh_y, error_y],
          titles        = [r'$u^{ex}_y(x,y)$', r'$u^h_y(x,y)$', r'$|(u^{ex}-u^h)_y(x,y)|$'],
          surface_plot  = True
    )
