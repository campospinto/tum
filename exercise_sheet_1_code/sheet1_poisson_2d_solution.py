import time
import numpy as np

from sympde.calculus    import dot, grad
from sympde.expr        import BilinearForm, LinearForm, integral
from sympde.topology    import elements_of, Square
from sympde.topology    import ScalarFunctionSpace

from psydac.api.discretization  import discretize
from psydac.api.settings        import PSYDAC_BACKEND_GPYCCEL
from psydac.fem.basic           import FemField
from psydac.linalg.basic        import IdentityOperator
from psydac.linalg.solvers      import inverse

from sheet1_utils import DirichletLinearOperator_2D

#==============================================================================
def run_poisson_2d(phi_ex, f, domain, ncells, degree, bc, epsilon, solution=True):

    assert all([ncells[i] > 0 for i in range(2)])
    assert all([degree[i] > 0 for i in range(2)])
    # Choose either a penalization or a projection method to enforce boundary conditions
    assert bc == 'pen' or bc == 'pro'
    if bc == 'pen':
        assert epsilon > 0

    #+++++++++++++++++++++++++++++++
    # 1. Abstract model
    #+++++++++++++++++++++++++++++++

    V    = ScalarFunctionSpace('V', domain, kind='h1')
    u, v = elements_of(V, names='u, v')

    # Define the bilinear form defining the system matrix A
    if solution:
        a = BilinearForm((u, v), integral(domain, dot(grad(u), grad(v))))
    else:
        a = BilinearForm((u, v), integral(domain, u*v))

    # Define the bilinear form defining the mass matrix
    m = BilinearForm((u, v), integral(domain, u * v))

    # Define the linear form defining the right hand side vector f_vec
    l = LinearForm(v, integral( V.domain, f*v ))

    #+++++++++++++++++++++++++++++++
    # 2. Discretization
    #+++++++++++++++++++++++++++++++

    # Discretize the domain and the function space (->FEM space)
    domain_h    = discretize(domain, ncells=ncells, periodic=[False, False])
    V_h         = discretize(V, domain_h, degree=degree)
    
    # Discretize all (bi)linear forms
    a_h = discretize(a, domain_h, (V_h, V_h), backend=PSYDAC_BACKEND_GPYCCEL)
    l_h = discretize(l, domain_h, V_h, backend=PSYDAC_BACKEND_GPYCCEL)
    m_h = discretize(m, domain_h, (V_h, V_h), backend=PSYDAC_BACKEND_GPYCCEL)

    # Assemble all discrete (bi)linear forms
    A = a_h.assemble()
    f_vec  = l_h.assemble()
    M = m_h.assemble()

    # Enforce the homogeneous Dirichlet BCs. 
    I_0     = IdentityOperator(V_h.vector_space)
    # P_0 sets all boundary dofs to zero
    P_0     = DirichletLinearOperator_2D(I_0, V, V)
    # P_Gamma sets all interior dofs to zero
    P_Gamma = I_0 - P_0
    if bc == 'pen':
        A_bc        = (1/epsilon) * P_Gamma + A
        f_vec_bc    = f_vec
    elif bc == 'pro':
        A_bc        = P_0 @ A @ P_0 + P_Gamma @ M @ P_Gamma
        f_vec_bc    = P_0 @ f_vec
    else:
        raise ValueError('Unknown method to enforce boundary conditions')

    # Now we can solve for phi ...
    A_bc_solver = inverse(A_bc, 'cg', tol=1e-12, maxiter=1000)
    timing  = {}
    t0      = time.time()
    phi_h   = A_bc_solver @ f_vec_bc
    t1      = time.time()
    timing['solution'] = t1 - t0
    
    info = A_bc_solver.get_info()

    # ... and determine the L2 error
    # For that matter, we compute the coefficients of the L2 projection of phi_ex
    k       = LinearForm(v, integral(domain, phi_ex * v))
    k_h     = discretize(k, domain_h, V_h, backend=PSYDAC_BACKEND_GPYCCEL)
    k_vec   = k_h.assemble()
    M_inv = inverse(M, 'cg', tol=1e-12, maxiter=1000)
    phi_ex_coeffs = M_inv @ k_vec

    diff        = phi_h - phi_ex_coeffs
    t0          = time.time()
    l2_error    = np.sqrt( diff.dot( M @ diff ) )
    t1          = time.time()
    timing['diagnostics'] = t1 - t0

    phi_h_fun = FemField(V_h, phi_h)

    return phi_h_fun, info, timing, l2_error

#==============================================================================
# SCRIPT CAPABILITIES
#==============================================================================
if __name__ == '__main__':

    from sympy import pi, sin, lambdify
    from sheet1_utils import plot

    ### 2D solver for the Poisson equation on the unit square S = ]0,1[ x ]0,1[ 
    ### with homogeneous Dirichlet boundary conditions (BCs)

    ### - laplacian phi = f in S
    ###             phi = 0 on the boundary of S = Gamma

    ### We use the method of manufactured solutions to test our code
    ### We define the exact solution phi_ex
    ###     phi_ex(x,y) =           sin(2 * pi * x) * sin(2 * pi * y)
    ### and compute the right hand side f
    ###     f(x,y)      = 8 * pi² * sin(2 * pi * x) * sin(2 * pi * y)

    ### The aim is to solve for phi using f

    domain  = Square('S', bounds1=(0, 1), bounds2=(0, 1))
    x,y     = domain.coordinates
    phi_ex  = sin(2*pi*x) * sin(2*pi*y)
    f       = 8 * (pi**2) * sin(2 * pi * x) * sin(2 * pi * y)

    ne      = [8, 8]    # number of Bspline cells
    degree  = [2, 2]    # Bspline degree
    bc      = 'pro'     # 'pen' or 'pro'
    epsilon = 1e-20     # 1/epsilon is the penalization parameter

    phi_h_fun, info, timing, l2_error = run_poisson_2d(phi_ex, f, domain, ne, degree, bc, epsilon)

    # ...
    print( '> Grid          :: [{ne1},{ne2}]'.format( ne1=ne[0], ne2=ne[1]) )
    print( '> Degree        :: [{p1},{p2}]'  .format( p1=degree[0], p2=degree[1] ) )
    print( '> CG info       :: ',info )
    print( '> L2 error      :: {:.2e}'.format( l2_error ) )
    print( '' )
    print( '> Solution time :: {:.2e}'.format( timing['solution'] ) )
    print( '> Evaluat. time :: {:.2e}'.format( timing['diagnostics'] ) )
    
    phi_ex_fun = lambdify(domain.coordinates, phi_ex)
    error = lambda x, y: abs(phi_ex_fun(x, y) - phi_h_fun(x, y))

    plot(gridsize_x   = 100, 
        gridsize_y    = 100, 
        title         = r'approximation of solution $\phi$', 
        funs          = [phi_ex_fun, phi_h_fun, error], 
        titles        = [r'$\phi_{ex}(x,y)$', r'$\phi_h(x,y)$', r'$|(\phi_{ex}-\phi_h)(x,y)|$'],
        surface_plot  = False
    )
