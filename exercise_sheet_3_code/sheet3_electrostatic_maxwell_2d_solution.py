import time
import numpy as np

from sympy  import pi, sin, cos, Tuple

from sympde.calculus    import dot
from sympde.expr.expr   import BilinearForm, integral
from sympde.topology    import Derham, elements_of, Square

from psydac.api.discretization      import discretize
from psydac.api.settings            import PSYDAC_BACKEND_GPYCCEL
from psydac.fem.basic               import FemField
from psydac.linalg.basic            import IdentityOperator
from psydac.linalg.solvers          import inverse

from sheet3_utils import DirichletLinearOperator_2D, get_M1_block_kron_solver_2D

#==============================================================================
def run_electrostatic_maxwell_2d(Eex, rho, domain, ncells, degree):

    #++++++++++++++++++++++++++++++++++++++++++++++
    # 1. Definition of bilinear forms required
    #++++++++++++++++++++++++++++++++++++++++++++++

    derham = Derham(domain, sequence=['h1', 'hcurl', 'l2'])

    V0 = derham.V0
    V1 = derham.V1
    V2 = derham.V2

    u0, v0 = elements_of(V0, names='u0, v0')
    u1, v1 = elements_of(V1, names='u1, v1')
    u2, v2 = elements_of(V2, names='u2, v2')

    m0 = BilinearForm((u0, v0), integral(domain,      u0*v0))
    m1 = BilinearForm((u1, v1), integral(domain,      dot(u1, v1)))
    m2 = BilinearForm((u2, v2), integral(domain,      u2 * v2))

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # 2. Definition of various linear and projection operators
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    domain_h = discretize(domain, ncells=ncells)
    derham_h = discretize(derham, domain_h, degree=degree)

    V0_h = derham_h.V0
    V1_h = derham_h.V1
    V2_h = derham_h.V2

    G, C = derham_h.derivatives_as_matrices

    P0, P1, P2 = derham_h.projectors()

    V1_vs       = V1_h.vector_space
    I1          = IdentityOperator(V1_vs)
    P1_0        = DirichletLinearOperator_2D(I1, V1, V1)
    P1_Gamma    = I1 - P1_0

    backend = PSYDAC_BACKEND_GPYCCEL

    m0_h = discretize(m0, domain_h, (V0_h, V0_h), backend=backend)
    m1_h = discretize(m1, domain_h, (V1_h, V1_h), backend=backend)
    m2_h = discretize(m2, domain_h, (V2_h, V2_h), backend=backend)

    M0 = m0_h.assemble()
    M1 = m1_h.assemble()
    M2 = m2_h.assemble()

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # 3. Assemble A_bc and rhs_bc, s.th. A_bc Eh_coeffs = rhs_bc
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
      
    G_0     = DirichletLinearOperator_2D(G, V0, V1)
    C_0     = DirichletLinearOperator_2D(C, V1, V2)
    M0_0    = DirichletLinearOperator_2D(M0, V0, V0, mass_matrix=True)
    M1_0    = DirichletLinearOperator_2D(M1, V1, V1, mass_matrix=True)

    Gt_0 = G_0.T
    Ct_0 = C_0.T

    M0_0_inv = inverse(M0_0, 'cg', tol=1e-11, maxiter=1000)

    D_0     = - M0_0_inv @ Gt_0 @ M1_0
    Dt_0    = - M1_0 @ G_0 @ M0_0_inv

    # SYSTEM MATRIX
    use_D_explicitely = False
    if use_D_explicitely == True:
        A = Ct_0 @ M2 @ C_0 + Dt_0 @ M0_0 @ D_0
    else:
        A = Ct_0 @ M2 @ C_0 + M1_0 @ G_0 @ M0_0_inv @ Gt_0 @ M1_0

    # RHS
    rho_lambdified  = lambdify(domain.coordinates, rho)
    rho_field       = P0(rho_lambdified)
    rho_coeffs      = rho_field.coeffs
    rhs             = - M1_0 @ G_0 @ rho_coeffs

    # PROJECTION METHOD
    A_bc    = P1_0 @ A @ P1_0 + P1_Gamma @ M1 @ P1_Gamma
    rhs_bc  = P1_0 @ rhs

    #A_bc_inv = inverse(A_bc, 'cg', tol=1e-9, maxiter=1000)
    M1_kron_solver = get_M1_block_kron_solver_2D(V1_vs, ncells, degree, periodic=[False, False])
    M1_0_solver = inverse(M1_0, 'pcg', pc=M1_kron_solver, tol=1e-11, maxiter=1000, recycle=False)
    A_bc_inv = inverse(A_bc, 'pcg', pc=M1_0_solver, tol=1e-10, maxiter=1000)

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # 4. Solve the system
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++

    timing      = {}
    t0          = time.time()
    Eh_coeffs   = A_bc_inv @ rhs_bc
    t1          = time.time()
    timing['solution_E'] = t1-t0

    rhoh_coeffs   = D_0 @ Eh_coeffs
    curlEh_coeffs = C_0 @ Eh_coeffs

    Eh      = FemField(V1_h, Eh_coeffs)
    rhoh    = FemField(V0_h, rhoh_coeffs)
    curlEh  = FemField(V2_h, curlEh_coeffs)

    info_E = A_bc_inv.get_info()

    Eex_x_lambdified    = lambdify(domain.coordinates, Eex[0])
    Eex_y_lambdified    = lambdify(domain.coordinates, Eex[1])
    Eex_lambdified      = (Eex_x_lambdified, Eex_y_lambdified)
    Eex_coeffs          = P1(Eex_lambdified).coeffs

    t0          = time.time()
    l2_error_E  = np.sqrt((Eex_coeffs - Eh_coeffs).dot(M1 @ (Eex_coeffs - Eh_coeffs)))
    t1          = time.time()
    timing['diagnostics_E'] = t1-t0

    return Eh, info_E, timing, l2_error_E, rhoh, curlEh

#==============================================================================
if __name__ == '__main__':

    from sympy              import lambdify
    from sheet3_utils       import plot

    domain = Square('S', bounds1=(0, 1), bounds2=(0, 1))
    x,y    = domain.coordinates

    Eex    = Tuple( (-1/pi)*cos(pi*x) * sin(pi*y),
                    (-1/pi)*cos(pi*y) * sin(pi*x))
    rho    = 2*sin(pi*x)*sin(pi*y)

    ne     = [32, 32]   # number of Bspline cells
    degree = [4, 4]   # Bspline degree

    Eh, info_E, timing, l2_error_E, rhoh, curlEh = run_electrostatic_maxwell_2d(Eex, rho, domain, ncells=ne, degree=degree)

    print( '> Convergence Information Electric Field E')
    print( '> Grid          :: [{ne1},{ne2}]'.format( ne1=ne[0], ne2=ne[1]) )
    print( '> Degree        :: [{p1},{p2}]'  .format( p1=degree[0], p2=degree[1] ) )
    print( '> CG info       :: ',info_E )
    print( '> L2 error      :: {:.2e}'.format( l2_error_E ) )
    print( '' )
    print( '> Solution time :: {:.2e}'.format( timing['solution_E'] ) )
    print( '> Evaluat. time :: {:.2e}'.format( timing['diagnostics_E'] ) )

    E_x         = lambdify(domain.coordinates, Eex[0])
    E_y         = lambdify(domain.coordinates, Eex[1])
    Eh_x        = Eh[0]
    Eh_y        = Eh[1]
    error_E_x   = lambda x, y: abs(E_x(x, y) - Eh_x(x,y))
    error_E_y   = lambda x, y: abs(E_y(x, y) - Eh_y(x,y))

    rho_lambdified  = lambdify(domain.coordinates, rho)
    error_rho       = lambda x, y: abs(rho_lambdified(x, y) - rhoh(x, y))

    plot(gridsize_x     = 100, 
         gridsize_y     = 100, 
         title          = r'approximation of $E$, $x$ component', 
         funs           = [E_x, Eh_x, error_E_x], 
         titles         = [r'$E^{ex}_x(x,y)$', r'$E^h_x(x,y)$', r'$|(E^{ex}-E^h)_x(x,y)|$'],
         surface_plot   = True
    )

    plot(gridsize_x     = 100, 
         gridsize_y     = 100, 
         title          = r'approximation of $E$, $y$ component', 
         funs           = [E_y, Eh_y, error_E_y], 
         titles         = [r'$E^{ex}_y(x,y)$', r'$E^h_y(x,y)$', r'$|(E^{ex}-E^h)_y(x,y)|$'],
         surface_plot   = True
    )

    plot(gridsize_x     = 100,
         gridsize_y     = 100,
         title          = r'approximation of $\rho$',
         funs           = [rho_lambdified, rhoh, error_rho],
         titles         = [r'$\rho^{ex}(x,y)$', r'$\rho^h(x,y)$', r'$|(\rho^{ex}-\rho^h)(x,y)|$'],
         surface_plot   = True
    )

    plot(gridsize_x     = 100,
         gridsize_y     = 100,
         title          = r'approximation of $curl E$',
         funs           = [curlEh, ],
         titles         = [r'$curl E^h(x,y)$',],
         surface_plot   = False
    )
