import numpy as np
import os

from scipy.sparse               import bmat, csc_matrix
from scipy.sparse.linalg        import inv
from scipy.sparse.linalg        import spsolve, eigsh

from sympde.calculus            import dot, cross
from sympde.expr                import BilinearForm, integral
from sympde.topology            import elements_of, Derham, Square, PolarMapping, NormalVector
from sympde.topology.basic      import Union
from sympde.utilities.utils     import plot_domain

from psydac.api.discretization  import discretize
from psydac.api.settings        import PSYDAC_BACKEND_GPYCCEL
from psydac.fem.basic           import FemField
from psydac.linalg.block        import BlockLinearOperator, BlockVectorSpace
from psydac.linalg.utilities    import array_to_psydac
from psydac.linalg.stencil      import StencilMatrix
from psydac.linalg.solvers      import inverse
from psydac.api.postprocessing  import OutputManager, PostProcessManager

def get_identity_matrix_2d(Vh):
    """
    As opposed to psydac.linalg.basic IdentityOperator, returns a non-matrix-free IdentityMatrix.
    Could be implemented inside IdentityOperator's tosparse() method.
    
    Parameters
    ----------
    Vh : psydac.fem.tensor.TensorFemSpace | psydac.fem.vector.VectorFemSpace
        belonging to a 2D de Rham sequence

    Returns
    -------
    I : BlockLinearOperator | StencilMatrix
        IdentityOperator with implemented tosparse() method.
    """

    Vh_vs  = Vh.vector_space

    block = isinstance(Vh_vs, BlockVectorSpace)

    if block:
        I = BlockLinearOperator(Vh_vs, Vh_vs)

        nblocks = Vh_vs.n_blocks

        for n in range(nblocks):
            Vh_vs_n = Vh_vs[n]
            I_nn = StencilMatrix(Vh_vs_n, Vh_vs_n)
            pad1, pad2 = Vh_vs_n.pads
            I_nn._data[pad1:-pad1, pad2:-pad2, pad1, pad2] = 1
            I[n,n] = I_nn

        return I
    else:
        I = StencilMatrix(Vh_vs, Vh_vs)
        pad1, pad2 = Vh_vs.pads
        I._data[pad1:-pad1, pad2:-pad2, pad1, pad2] = 1

        return I

def get_conforming_projections_2d(derham_h, sequence, periodic):
    """
    Returns two conforming projection operators belonging to the first two spaces of a 2d de Rham sequence.,
    cP0: V0h -> V0h
    cP1: V1h -> V1h
    mapping coefficients to coefficients corresponding to functions satisfying homogeneous DBCs.

    Parameters
    ----------
    derham_h : psydac.api.feec.DiscreteDerham
    sequence : list | tuple
        either ['h1', 'hcurl', 'l2'] or ['h1', 'hdiv', 'l2']
    periodic : list | tuple
        of length 2, indicating the periodicity of the domain

    Returns
    -------
    cP0 : StencilMatrix
    cP1 : BlockLinearOperator
    
    """

    assert len(periodic) == 2
    assert all([isinstance(p, bool) for p in periodic])

    if periodic == [True, True]:
        raise ValueError('! Conforming Projections are Identity Matrices if domain is periodic in each direction !')

    p1 = 1 if not periodic[0] else 0
    p2 = 1 if not periodic[1] else 0

    V0h             = derham_h.V0
    V0h_vs          = V0h.vector_space
    pad0_1, pad0_2  = V0h_vs.pads
    cP0             = StencilMatrix(V0h_vs, V0h_vs)
    cP0._data       [pad0_1+p1:-(pad0_1+p1), pad0_2+p2:-(pad0_2+p2), pad0_1, pad0_2] = 1

    
    V1h                 = derham_h.V1
    V1h_vs              = V1h.vector_space
    V1_1                = V1h_vs[0]
    pad11_1, pad11_2    = V1_1.pads
    cP1_1               = StencilMatrix(V1_1, V1_1)
    V1_2                = V1h_vs[1]
    pad12_1, pad12_2    = V1_2.pads
    cP1_2               = StencilMatrix(V1_2, V1_2)

    if sequence[1] == 'hcurl':
        cP1_1._data     [pad11_1:-pad11_1, pad11_2+p2:-(pad11_2+p2), pad11_1, pad11_2] = 1
        cP1_2._data     [pad12_1+p1:-(pad12_1+p1), pad12_2:-pad12_2, pad12_1, pad12_2] = 1
    elif sequence[1] == 'hdiv':
        cP1_1._data     [pad11_1+p1:-(pad11_1+p1), pad11_2:-pad11_2, pad11_1, pad11_2] = 1
        cP1_2._data     [pad12_1:-pad12_1, pad12_2+p2:-(pad12_2+p2), pad12_1, pad12_2] = 1
    else:
        raise ValueError(f"Sequence {sequence} must be either ['h1', 'hcurl', 'l2'] or ['h1', 'hdiv', 'l2'].")

    cP1                 = BlockLinearOperator(V1h_vs, V1h_vs, ((cP1_1, None),
                                                               (None, cP1_2)))

    return cP0, cP1

def get_eigenvalues(nb_eigs, sigma, A_m, M_m):
    """
    Compute the eigenvalues of the matrix A close to sigma and right-hand-side M
    Function seen and adapted from >>> psydac_dev/psydac/feec/multipatch/examples/hcurl_eigen_pbms_conga_2d.py <<< 
    (Commit a748a4d8c1569a8765f6688d228f65ea6073c252)

    Parameters
    ----------
    nb_eigs : int
        Number of eigenvalues to compute
    sigma : float
        Value close to which the eigenvalues are computed
    A_m : sparse matrix
        Matrix A
    M_m : sparse matrix
        Matrix M
    """

    print()
    print('-----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  ----- ')
    print(
        'computing {0} eigenvalues (and eigenvectors) close to sigma={1} with scipy.sparse.eigsh...'.format(nb_eigs, sigma))
    mode = 'normal'
    which = 'LM'
    ncv = 4 * nb_eigs
    max_shape_splu = 24000
    if A_m.shape[0] >= max_shape_splu:
        raise ValueError(f'Matrix too large.')
        
    eigenvalues, eigenvectors = eigsh(
        A_m, k=nb_eigs, M=M_m, sigma=sigma, mode=mode, which=which, ncv=ncv)

    print("done: eigenvalues found: " + repr(eigenvalues))
    print('-----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----')
    print()

    return eigenvalues, eigenvectors

#==============================================================================
def run_magnetostatic_maxwell_2d(A_ex, b_ex, domain, ncells, degree, periodic):
    """
    Solver for the 2D magnetostatic problem:
    Find A in V1h satisfying

            div A       = 0
            curl curl A = J

    written in the form of a mixed problem: find s in H_0^1, A in H_0(curl), such that

        - s + G^* A = 0      in \\Omega
        G s + CC  A = J      in \\Omega

    with operators

        G  :   v -> grad v
        G^*:   u -> -div u
        CC :   u -> curl curl u

    Then the curl of the solution A = (Ax, Ay), i.e., B = curl A, satisfies the original magnetostatic equation curl B = J.

    Here the operators G and CC are discretized with

        Gh: V0h -> V1h  and  CCh: V1h -> V1h

    in a FEEC approach involving a discrete sequence on a 2D singlepatch Annulus,

        V0h  --grad->  V1h  -—curl-> V2h

    with homogeneous Dirichlet boundary conditions.

    Harmonic constraint: As dim_harmonic_space = 1, a constraint is added, of the form

        u in H^\\perp

    where H = ker(L) is the kernel of the Hodge-Laplace operator L = curl curl - grad div

    """

    # Set Psydac backend
    backend = PSYDAC_BACKEND_GPYCCEL

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

    # Homogeneous DBCs do not have to be satisfied on "periodic part of the boundary"
    boundary = Union(domain.get_boundary(0,-1), domain.get_boundary(0,1))
    nn       = NormalVector('nn')

    # Measure the extend to which a V1h function satisfies the homogeneous DBCs
    b1 = BilinearForm((u1, v1), integral(boundary, cross(nn, u1) * cross(nn, v1)))

    # Mass Matrices
    m0 = BilinearForm((u0, v0), integral(domain, u0*v0))
    m1 = BilinearForm((u1, v1), integral(domain, dot(u1, v1)))
    m2 = BilinearForm((u2, v2), integral(domain, u2*v2))

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # 2. Definition of various linear and projection operators related to the de Rham sequence
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    domain_h = discretize(domain, ncells=ncells, periodic=[False, True])
    derham_h = discretize(derham, domain_h, degree=degree)

    V0h = derham_h.V0
    V1h = derham_h.V1
    V2h = derham_h.V2

    # Gradient and Curl Matrix
    G, C = derham_h.derivatives_as_matrices

    # Non-conforming projection operators (not respecting homogeneous DBCs)
    P0, P1, P2 = derham_h.projectors()

    # conforming Psydac.LinearOperator projection operators (respecting homogeneous DBCs) for V0 and V1
    cP0, cP1    = get_conforming_projections_2d(derham_h, ['h1', 'hcurl', 'l2'], periodic)

    # StencilMatrix Identity Matrices for V0 and V1
    #I0, I1      = get_stencil_identity_matrix_2D(derham_h, ['h1', 'hcurl', 'l2'], ncells, degree, periodic)
    I0          = get_identity_matrix_2d(V0h)
    I1          = get_identity_matrix_2d(V1h)

    J0          = I0 - cP0
    J1          = I1 - cP1

    b1h = discretize(b1, domain_h, (V1h, V1h), backend=backend)

    m0h = discretize(m0, domain_h, (V0h, V0h), backend=backend)
    m1h = discretize(m1, domain_h, (V1h, V1h), backend=backend)
    m2h = discretize(m2, domain_h, (V2h, V2h), backend=backend)

    B1          = b1h.assemble()

    M0          = m0h.assemble()
    M1          = m1h.assemble()
    M2          = m2h.assemble()

    # Get sparse representation of operators involved in computing the eigenvector corresponding to the eigenvalue 0.

    cP0_m       = cP0.tosparse()
    cP1_m       = cP1.tosparse()
    J1_m        = J1.tosparse()
    J0_m        = J0.tosparse()
    M0_m        = M0.tosparse()
    M1_m        = M1.tosparse()
    M2_m        = M2.tosparse()
    G_m         = G.tosparse()
    C_m         = C.tosparse()

    # To work only within the space of functions satisfying the right boundary conditions, we alter certain linear operators
    G           = G @ cP0
    C           = C @ cP1
    M0          = cP0 @ M0 @ cP0 + J0 @ M0 @ J0
    M1          = cP1 @ M1 @ cP1 + J1 @ M1 @ J1
    M0_inv      = inverse(M0, 'cg', maxiter=1000, tol=1e-9)

    G_m         = G_m @ cP0_m
    C_m         = C_m @ cP1_m
    M0_m        = cP0_m @ M0_m @ cP0_m + J0_m @ M0_m @ J0_m
    M1_m        = cP1_m @ M1_m @ cP1_m + J1_m @ M1_m @ J1_m
    inv_M0_m    = inv(M0_m.tocsc())
    inv_M0_m.eliminate_zeros()

    # Stabilization Matrix
    S1_m        = J1_m @ M1_m @ J1_m
    reg_S1_m    = 10. * S1_m

    # Building the components of the discrete Hodge-Laplace operator
    tG_m        = M1_m @ G_m
    CC_m        = C_m.transpose() @ M2_m @ C_m
    GD_m        = - tG_m @ inv_M0_m @ G_m.transpose() @ M1_m

    # The discrete Hodge-Laplace operator in sparse format plus a stabilization matrix
    L_m         = CC_m - GD_m + reg_S1_m

    # Compute the eigenvector corresponding to the eigenvalue 0 for the harmonic constraint
    dim_harmonic_space = 1

    eigenvalues, eigenvectors = get_eigenvalues(dim_harmonic_space + 1, 1e-6, L_m, M1_m)

    # For visualization in Paraview, express the harmonic field as a FemField
    ev = array_to_psydac(eigenvectors[:,0], V1h.vector_space)
    Ev = FemField(V1h, ev)

    hf_cs = eigenvectors[:,0]
    hf_m  = csc_matrix(hf_cs).transpose()
    MH_m  = M1_m @ hf_m
    
    # Build the Stiffness Matrix (reg_S1_m for stabilization - it does not affect the result)
    A_m = bmat([[M0_m, tG_m.transpose(), None],
                [tG_m, CC_m + reg_S1_m,  MH_m],
                [None, MH_m.transpose(), None]])

    # Build the rhs = [rhs1_c, rhs2_c, rhs3_c]

    # First     component of rhs
    rhs1_c = np.zeros(V0h.nbasis)

    # Second    component of rhs
    A_ex    = P1(A_ex)
    A_ex_c  = A_ex.coeffs
    rhs2    = C.T @ M2 @ C @ A_ex_c
    rhs2_c  = rhs2.toarray()

    # Third     component of rhs
    rhs3_c = np.zeros(dim_harmonic_space)

    # Full rhs
    rhs_c = np.block([rhs1_c, rhs2_c, rhs3_c])

    # direct solve with scipy spsolve ------------------------------
    sol_c = spsolve(A_m.asformat('csr'), rhs_c)
    #   ------------------------------------------------------------

    sh_c = sol_c[:V0h.nbasis]
    Ah_c = sol_c[V0h.nbasis:V0h.nbasis + V1h.nbasis]
    ph_c = sol_c[V0h.nbasis+V1h.nbasis:]

    sh_c = array_to_psydac(sh_c, V0h.vector_space)
    Ah_c = array_to_psydac(Ah_c, V1h.vector_space)

    divAh_c         = M0_inv @ G.T @ M1 @ Ah_c
    Ah_boundary     = Ah_c.dot( B1 @ Ah_c )
    A_ex_boundary   = A_ex_c.dot( B1 @ A_ex_c )
    bh_c            = C @ Ah_c

    Ah              = FemField(V1h, Ah_c)
    b_ex            = P2(b_ex)
    b_ex_c          = b_ex.coeffs
    bh              = FemField(V2h, bh_c)

    Ah_norm         = Ah_c.dot(M1 @ Ah_c)
    bh_norm         = bh_c.dot(M2 @ bh_c)
    sh_norm         = sh_c.dot(M0 @ sh_c)
    divAh_norm      = divAh_c.dot(M0 @ divAh_c)

    A_norm          = A_ex_c.dot(M1 @ A_ex_c)
    b_ex_norm       = b_ex_c.dot(M2 @ b_ex_c)
    
    A_diff          = A_ex_c - Ah_c
    b_diff          = b_ex_c - bh_c

    A_diff_norm     = A_diff.dot(M1 @ A_diff)
    b_diff_norm     = b_diff.dot(M2 @ b_diff)

    print(f'> ---------- Vector Potential A ---------- <')
    print(f'||   A x n   ||_boundary   = {A_ex_boundary}')
    print(f'||   Ah x n  ||_boundary   = {Ah_boundary}')
    print(f'||     s     ||            = {sh_norm}')
    print(f'||  div(Ah)  ||            = {divAh_norm}')
    print(f'||     A     ||            = {A_norm}')
    print(f'||     Ah    ||            = {Ah_norm}')
    print(f'||   A - Ah  ||            = {A_diff_norm}')
    print()

    print(f'> ---------- Magnetic Field B ---------- <')
    print(f'||     B     ||            = {b_ex_norm}')
    print(f'||     Bh    ||            = {bh_norm}')
    print(f'||   B - Bh  ||            = {b_diff_norm}')
    print()

    print(f'> ---------- Lagrange Multiplier p ---------- <')
    print(f'      ph_c                 = {ph_c}')
    print()

    return V1h, V2h, Ah, A_ex, Ev, bh, b_ex

#==============================================================================
if __name__ == '__main__':

    # Set rmin and rmax defining the Annulus
    rmin, rmax = 0.3, 1.

    domain_log = Square('Square', bounds1=(0., 1.), bounds2=(0., 2*np.pi))
    F = PolarMapping('Annulus', dim=2, c1=0., c2=0., rmin=rmin, rmax=rmax)
    domain = F(domain_log)
    plot_domain(domain, draw=True, isolines=True)

    # Define exact solution A_ex for method of manufactured solution
    r    = lambda x, y : np.sqrt(x**2 + y**2)
    A_1  = lambda x, y : (x*y) / (r(x,y)**3)
    A_2  = lambda x, y : (y**2) / (r(x,y)**3)
    A_ex = (A_1, A_2)

    b_ex = lambda x, y : -1 * ( (x*y**2 + x**3) / (r(x,y)**5) )

    ncells      = [16, 16]      # number of Bspline cells
    degree      = [3, 3]        # Bspline degree
    periodic    = [False, True] # periodicity of the domain

    V1h, V2h, Ah, A_ex, Ev, bh, b_ex = run_magnetostatic_maxwell_2d(A_ex, b_ex, domain, ncells=ncells, degree=degree, periodic=periodic)

    # ------------------------------

    # Save the results using OutputManager

    os.makedirs('V1', exist_ok=True)
    os.makedirs('V2', exist_ok=True)

    Om = OutputManager(
        f'V1/space_info_{domain.name}',
        f'V1/field_info_{domain.name}',   
    )
    Om.add_spaces(V=V1h)
    Om.export_space_info()
    Om.set_static()

    Om.export_fields(Ah     = Ah)
    Om.export_fields(A_ex   = A_ex)
    Om.export_fields(Ev     = Ev)

    Om.close()

    Om = OutputManager(
        f'V2/space_info_{domain.name}',
        f'V2/field_info_{domain.name}', 
    )
    Om.add_spaces(V=V2h)
    Om.export_space_info()
    Om.set_static()

    Om.export_fields(bh     = bh)
    Om.export_fields(b_ex   = b_ex)

    Om.close()

    # Export the results to VTK using PostProcessManager

    Pm = PostProcessManager(
        domain=domain,
        space_file=f'V1/space_info_{domain.name}.yaml',
        fields_file=f'V1/field_info_{domain.name}.h5', 
    )

    Pm.export_to_vtk(
        f'V1/A_{domain.name}',
        grid=None,
        npts_per_cell=3,
        fields=('Ah','A_ex')
    )

    Pm.export_to_vtk(
        f'V1/H_{domain.name}',
        grid=None,
        npts_per_cell=3,
        fields=('Ev')
    )

    Pm.close()

    Pm = PostProcessManager(
        domain=domain,
        space_file=f'V2/space_info_{domain.name}.yaml',
        fields_file=f'V2/field_info_{domain.name}.h5',  
    )

    Pm.export_to_vtk(
        f'V2/B_{domain.name}',
        grid=None,
        npts_per_cell=3,
        fields=('bh', 'b_ex')
    )

    Pm.close()
