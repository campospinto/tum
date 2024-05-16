import time
import numpy as np

from sympde.calculus    import dot
from sympde.expr        import BilinearForm, LinearForm, integral
from sympde.topology    import Derham, elements_of, Square

from psydac.api.discretization  import discretize
from psydac.api.settings        import PSYDAC_BACKEND_GPYCCEL
from psydac.fem.basic           import FemField
from psydac.linalg.basic        import IdentityOperator
from psydac.linalg.solvers      import inverse

from sheet3_utils import DirichletLinearOperator_2D

#==============================================================================
def run_poisson_2d(phi_ex, f, domain, ncells, degree, bc, epsilon):

    assert all([ncells[i] > 0 for i in range(2)])
    assert all([degree[i] > 0 for i in range(2)])
    # Choose either a penalization or a projection method to enforce boundary conditions
    assert bc == 'pen' or bc == 'pro'
    if bc == 'pen':
        assert epsilon > 0

    #++++++++++++++++++++++++++++++++++++++++++++++
    # 1. Definition of bilinear forms required
    #++++++++++++++++++++++++++++++++++++++++++++++

    derham = Derham(domain, sequence=['hi', 'hcurl', 'l2'])
    V0 = derham.V0
    V1 = derham.V1

    u0, v0 = elements_of(V0, names='u0, v0')
    u1, v1 = elements_of(V1, names='u1, v1')

    m0 = BilinearForm((u0, v0), integral(domain, u0 * v0))
    m1 = BilinearForm((u1, v1), integral(domain, dot(u1, v1)))

    l = LinearForm(v0, integral( domain, f*v0 ))

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # 2. Obtain discrete function spaces, projection operators 
    #    and derivative matrices from DiscreteDerham object
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Discretize the domain and the function space (->FEM space)
    domain_h    = discretize(domain, ncells=ncells)
    derham_h    = discretize(derham, domain_h, degree=degree)

    V0_h        = derham_h.V0
    V1_h        = derham_h.V1

    V0_h_vs     = V0_h.vector_space

    G, C = derham_h.derivatives_as_matrices

    P0, P1, P2 = derham_h.projectors()
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++
    # 3. Assemble the Linear System Ax = b (disregarding the boundary condition)
    #++++++++++++++++++++++++++++++++++++++++++++++++++

    backend = PSYDAC_BACKEND_GPYCCEL

    m0_h = discretize(m0, domain_h, (V0_h, V0_h), backend=backend)
    m1_h = discretize(m1, domain_h, (V1_h, V1_h), backend=backend)

    l_h = discretize(l, domain_h, V0_h, backend=backend)

    M0 = m0_h.assemble()
    M1 = m1_h.assemble()

    A = G.T @ M1 @ G

    f_vec = l_h.assemble()

    #++++++++++++++++++++++++++++++++++++++++++
    # 4. Enforce homogeneous Dirichlet BCs
    #++++++++++++++++++++++++++++++++++++++++++

    I_0     = IdentityOperator(V0_h_vs)
    P_0     = DirichletLinearOperator_2D(I_0, V0, V0)
    P_Gamma = I_0 - P_0

    if bc == 'pen':
        A_bc        = (1/epsilon) * P_Gamma + A
        f_vec_bc    = f_vec
    elif bc == 'pro':
        A_bc        = P_0 @ A @ P_0 + P_Gamma @ M0 @ P_Gamma
        f_vec_bc    = P_0 @ f_vec
    else:
        raise ValueError('Unknown method to enforce boundary conditions')

    #++++++++++++++++++++++++++++++++++++++++++++++++++
    # 5. Solve the Linear System A_bc x = f_vec_bc
    #++++++++++++++++++++++++++++++++++++++++++++++++++
    
    A_bc_solver = inverse(A_bc, 'cg', tol=1e-9, maxiter=1000)

    timing          = {}
    t0              = time.time()
    phi_h_coeffs    = A_bc_solver @ f_vec_bc
    t1              = time.time()
    timing['solution'] = t1 - t0
    
    phi_h   = FemField(V0_h, phi_h_coeffs)
    info    = A_bc_solver.get_info()

    from sympy  import lambdify
    phi_ex_lambdified   = lambdify(domain.coordinates, phi_ex)
    phi_ex_coeffs       = P0(phi_ex_lambdified).coeffs

    diff        = phi_h_coeffs - phi_ex_coeffs
    t0          = time.time()
    l2_error    = np.sqrt( diff.dot( M0 @ diff ) )
    t1          = time.time()
    timing['diagnostics'] = t1 - t0

    return phi_h, info, timing, l2_error

#==============================================================================
# SCRIPT CAPABILITIES
#==============================================================================
if __name__ == '__main__':

    from sympy import pi, sin, lambdify
    from sheet3_utils import plot

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

    ne      = [16, 16]  # number of Bspline cells
    degree  = [3, 3]    # Bspline degree
    bc      = 'pro'     # 'pen' or 'pro'
    epsilon = 1e-20     # 1/epsilon is the penalization parameter

    phi_h_fun, info, timing, l2_error = run_poisson_2d(phi_ex, f, domain, ne, degree, bc, epsilon)

    print( '> Grid          :: [{ne1},{ne2}]'.format( ne1=ne[0], ne2=ne[1]) )
    print( '> Degree        :: [{p1},{p2}]'  .format( p1=degree[0], p2=degree[1] ) )
    print( '> CG info       :: ',info )
    print( '> L2 error      :: {:.2e}'.format( l2_error ) )
    print( '' )
    print( '> Solution time :: {:.2e}'.format( timing['solution'] ) )
    print( '> Evaluat. time :: {:.2e}'.format( timing['diagnostics'] ) )
    
    phi_ex_fun  = lambdify(domain.coordinates, phi_ex)
    error       = lambda x, y: abs(phi_ex_fun(x, y) - phi_h_fun(x, y))

    plot(gridsize_x     = 100, 
         gridsize_y     = 100, 
         title          = r'approximation of solution $\phi$', 
         funs           = [phi_ex_fun, phi_h_fun, error], 
         titles         = [r'$\phi_{ex}(x,y)$', r'$\phi_h(x,y)$', r'$|(\phi_{ex}-\phi_h)(x,y)|$'],
         surface_plot   = True
    )
