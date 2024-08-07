import time
import numpy as np

from sympy  import pi, sin, cos, Tuple

from sympde.calculus      import dot, cross
from sympde.expr.expr     import LinearForm, BilinearForm, integral
from sympde.topology      import Derham, elements_of, NormalVector, Square

from psydac.api.discretization       import discretize
from psydac.api.settings             import PSYDAC_BACKEND_GPYCCEL
from psydac.fem.basic                import FemField
from psydac.linalg.solvers           import inverse

#==============================================================================
def run_maxwell_2d(Eex, f, alpha, domain, ncells, degree, kappa):

    #++++++++++++++++++++++++++++++++++++++++++++++
    # 1. Definition of bilinear forms required
    #++++++++++++++++++++++++++++++++++++++++++++++

    derham = Derham(domain, sequence=['h1', 'hcurl', 'l2'])

    V1 = derham.V1
    V2 = derham.V2

    u1, v1 = elements_of(V1, names='u1, v1')
    u2, v2 = elements_of(V2, names='u1, v1')

    nn       = NormalVector('nn')
    boundary = domain.boundary

    m1   = BilinearForm((u1, v1), integral(domain,      dot(u1, v1)))
    m2   = BilinearForm((u2, v2), integral(domain,      u2 * v2))
    m1_b = BilinearForm((u1, v1), integral(boundary,    cross(nn, u1) * cross(nn, v1)))

    l    = LinearForm(v1, integral(domain, dot(f, v1)))

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # 2. Obtain discrete function spaces, projection operators 
    #    and derivative matrices from DiscreteDerham object
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    domain_h = discretize(domain, ncells=ncells)
    derham_h = discretize(derham, domain_h, degree=degree)

    V1_h = derham_h.V1
    V2_h = derham_h.V2

    G, C = derham_h.derivatives_as_matrices

    P0, P1, P2 = derham_h.projectors()

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # 3. Assemble and Solve the Linear System Ax = f_vec
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++

    backend = PSYDAC_BACKEND_GPYCCEL

    m1_h    = discretize(m1,    domain_h, (V1_h, V1_h), backend=backend)
    m1_b_h  = discretize(m1_b,  domain_h, (V1_h, V1_h), backend=backend)
    m2_h    = discretize(m2,    domain_h, (V2_h, V2_h), backend=backend)

    l_h     = discretize(l,     domain_h, V1_h,         backend=backend)

    M1      = m1_h.assemble()
    M1_b    = m1_b_h.assemble()
    M2      = m2_h.assemble()

    A = C.T @ M2 @ C + alpha * M1 + kappa * M1_b

    from sheet3_utils import get_diagonal
    inv_diag_A = get_diagonal(A, inv=True)
    A_inv = inverse(A, 'pcg', pc=inv_diag_A, tol=1e-8)

    f_vec = l_h.assemble()

    timing      = {}
    t0          = time.time()
    Eh_coeffs   = A_inv @ f_vec
    t1          = time.time()
    timing['solution'] = t1-t0

    Eh      = FemField(V1_h, Eh_coeffs)
    info    = A_inv.get_info()

    Eex_x           = lambdify(domain.coordinates, Eex[0])
    Eex_y           = lambdify(domain.coordinates, Eex[1])
    Eex_lambdified  = (Eex_x, Eex_y)
    Eex_coeffs      = P1(Eex_lambdified).coeffs

    t0          = time.time()
    l2_error    = np.sqrt((Eex_coeffs - Eh_coeffs).dot(M1 @ (Eex_coeffs - Eh_coeffs)))
    t1          = time.time()
    timing['diagnostics'] = t1-t0

    return Eh, info, timing, l2_error

#==============================================================================
if __name__ == '__main__':

    from sympy              import lambdify
    from sheet3_utils       import plot

    domain = Square('S', bounds1=(0, 1), bounds2=(0, 1))
    x,y    = domain.coordinates
    omega  = 1.5
    alpha  = -omega**2
    Eex    = Tuple(sin(pi*y), sin(pi*x)*cos(pi*y))
    f      = Tuple(alpha*sin(pi*y) - pi**2*sin(pi*y)*cos(pi*x) + pi**2*sin(pi*y),
                  alpha*sin(pi*x)*cos(pi*y) + pi**2*sin(pi*x)*cos(pi*y))

    ne     = [8, 8]   # number of Bspline cells
    degree = [2, 2]   # Bspline degree
    kappa  = 1e6      # penalization parameter

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
         gridsize_y     = 100, 
         title          = r'approximation of solution $E$, $x$ component', 
         funs           = [Eex_x, Eh_x, error_x], 
         titles         = [r'$E^{ex}_x(x,y)$', r'$E^h_x(x,y)$', r'$|(E^{ex}-E^h)_x(x,y)|$'],
         surface_plot   = True
    )
    
    plot(gridsize_x     = 100,
         gridsize_y     = 100,
         title          = r'approximation of solution $E$, $y$ component',
         funs           = [Eex_y, Eh_y, error_y],
         titles         = [r'$E^{ex}_y(x,y)$', r'$E^h_y(x,y)$', r'$|(E^{ex}-E^h)_y(x,y)|$'],
         surface_plot   = True
    )
