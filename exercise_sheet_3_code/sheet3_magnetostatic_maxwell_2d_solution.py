import time
import numpy as np

from sympy  import pi, sin, cos, Tuple

from sympde.calculus      import dot
from sympde.expr.expr     import BilinearForm, integral
from sympde.topology      import Derham, elements_of, Square

from psydac.api.discretization       import discretize
from psydac.api.settings             import PSYDAC_BACKEND_GPYCCEL
from psydac.fem.basic                import FemField
from psydac.linalg.solvers           import inverse

from sheet3_utils import DirichletLinearOperator_2D

#==============================================================================
def run_magnetostatic_maxwell_2d(Bex, J, domain, ncells, degree):

    #++++++++++++++++++++++++++++++++++++++++++++++
    # 1. Definition of bilinear forms required
    #++++++++++++++++++++++++++++++++++++++++++++++

    derham = Derham(domain, sequence=['h1', 'hcurl', 'l2'])

    V1 = derham.V1
    V2 = derham.V2

    u1, v1  = elements_of(V1, names='u1, v1')
    u2, v2  = elements_of(V2, names='u1, v1')

    m1  = BilinearForm((u1, v1), integral(domain,      dot(u1, v1)))
    m2  = BilinearForm((u2, v2), integral(domain,      u2 * v2))

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # 2. Definition of various linear and projection operators
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    domain_h = discretize(domain, ncells=ncells)
    derham_h = discretize(derham, domain_h, degree=degree)

    V1_h = derham_h.V1
    V2_h = derham_h.V2

    G, C = derham_h.derivatives_as_matrices

    P0, P1, P2 = derham_h.projectors()

    backend = PSYDAC_BACKEND_GPYCCEL

    m1_h    = discretize(m1,    domain_h, (V1_h, V1_h), backend=backend)
    m2_h    = discretize(m2,    domain_h, (V2_h, V2_h), backend=backend)

    M1  = m1_h.assemble()
    M2  = m2_h.assemble()

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # 3. Assemble A and rhs, s.th. A Bh_coeffs = rhs
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++

    C_0     = DirichletLinearOperator_2D(C, V1, V2)
    M1_0    = DirichletLinearOperator_2D(M1, V1, V1, mass_matrix=True)

    Ct_0 = C_0.T

    M1_0_inv = inverse(M1_0, 'cg', tol=1e-9, maxiter=1000)

    vec_C_0     = M1_0_inv @ Ct_0 @ M2
    vec_Ct_0    = M2 @ C_0 @ M1_0_inv

    # SYSTEM MATRIX
    #A = vec_Ct_0 @ M1_0 @ vec_C_0
    A = M2 @ C_0 @ M1_0_inv @ Ct_0 @ M2

    # RHS
    J_x             = lambdify(domain.coordinates, J[0])
    J_y             = lambdify(domain.coordinates, J[1])
    J_lambdified    = (J_x, J_y)
    J_field         = P1(J_lambdified)
    J_coeffs        = J_field.coeffs
    rhs             = M2 @ C_0 @ J_coeffs

    A_inv = inverse(A, 'cg', tol=1e-9, maxiter=1000)

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # 4. Solve the system
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++

    timing      = {}
    t0          = time.time()
    Bh_coeffs   = A_inv @ rhs
    t1          = time.time()
    timing['solution_B'] = t1-t0

    Jh_coeffs   = vec_C_0 @ Bh_coeffs
    Bh          = FemField(V2_h, Bh_coeffs)
    Jh          = FemField(V1_h, Jh_coeffs)

    info_B = A_inv.get_info()

    Bex_lambdified  = lambdify(domain.coordinates, Bex)
    Bex_coeffs      = P2(Bex_lambdified).coeffs

    t0          = time.time()
    l2_error_B  = np.sqrt((Bex_coeffs - Bh_coeffs).dot(M2 @ (Bex_coeffs - Bh_coeffs)))
    t1          = time.time()
    timing['diagnostics_B'] = t1-t0

    return Bh, Jh, info_B, timing, l2_error_B

#==============================================================================
if __name__ == '__main__':

    from sympy              import lambdify
    from sheet3_utils       import plot

    domain = Square('S', bounds1=(0, 1), bounds2=(0, 1))
    x,y    = domain.coordinates
    Bex    = (1/pi) * cos(pi*x) * cos(pi*y)
    J      = Tuple(-cos(pi*x) * sin(pi*y), 
                    cos(pi*y) * sin(pi*x))

    ne     = [16, 16]   # number of Bspline cells
    degree = [3, 3]   # Bspline degree

    Bh, Jh, info_B, timing, l2_error_B = run_magnetostatic_maxwell_2d(Bex, J, domain, ncells=ne, degree=degree)

    print( '> Convergence Information Magnetic Field B')
    print( '> Grid          :: [{ne1},{ne2}]'.format( ne1=ne[0], ne2=ne[1]) )
    print( '> Degree        :: [{p1},{p2}]'  .format( p1=degree[0], p2=degree[1] ) )
    print( '> CG info       :: ',info_B )
    print( '> L2 error      :: {:.2e}'.format( l2_error_B ) )
    print( '' )
    print( '> Solution time :: {:.2e}'.format( timing['solution_B'] ) )
    print( '> Evaluat. time :: {:.2e}'.format( timing['diagnostics_B'] ) )
    
    Bex_lambdified = lambdify(domain.coordinates, Bex)
    error_B = lambda x, y: abs(Bex_lambdified(x, y) - Bh(x, y))

    plot(gridsize_x     = 100,
         gridsize_y     = 100,
         title          = r'approximation of solution $B$',
         funs           = [Bex_lambdified, Bh, error_B],
         titles         = [r'$B^{ex}(x,y)$', r'$B^h(x,y)$', r'$|(B^{ex}-B^h)(x,y)|$'],
         surface_plot   = True
    )

    J_x     = lambdify(domain.coordinates, J[0])
    J_y     = lambdify(domain.coordinates, J[1])
    Jh_x    = Jh[0]
    Jh_y    = Jh[1]
    error_x = lambda x, y: abs(J_x(x, y) - Jh_x(x, y))
    error_y = lambda x, y: abs(J_y(x, y) - Jh_y(x, y))

    plot(gridsize_x     = 100, 
          gridsize_y    = 100, 
          title         = r'approximation of $J$, $x$ component', 
          funs          = [J_x, Jh_x, error_x], 
          titles        = [r'$J^{ex}_x(x,y)$', r'$J^h_x(x,y)$', r'$|(J^{ex}-J^h)_x(x,y)|$'],
          surface_plot  = False
    )
    
    plot(gridsize_x     = 100,
          gridsize_y    = 100,
          title         = r'approximation of $J$, $y$ component',
          funs          = [J_y, Jh_y, error_y],
          titles        = [r'$J^{ex}_y(x,y)$', r'$J^h_y(x,y)$', r'$|(J^{ex}-J^h)_y(x,y)|$'],
          surface_plot  = False
    )
