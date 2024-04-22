import numpy                as np
import matplotlib.pyplot    as plt
from   matplotlib           import cm, colors

from sympde.expr                import EssentialBC
from sympde.topology.datatype   import H1Space, HcurlSpace, L2Space
from sympde.topology            import element_of

from psydac.linalg.basic        import LinearOperator
from psydac.linalg.block        import BlockVectorSpace
from psydac.api.essential_bc    import apply_essential_bc

def plot(gridsize_x, gridsize_y, title, funs, titles=None, surface_plot=False):

    x = np.linspace(0, 1, gridsize_x+1)
    y = np.linspace(0, 1, gridsize_y+1)
    xx, yy = np.meshgrid(x, y)
    vals = [[] for fun in funs]
    for i, fun in enumerate(funs):
        for xi, yi in zip(xx, yy):
            vals[i].append([fun(xii, yii) for xii, yii in zip(xi, yi)])
        vals[i] = np.array(vals[i])

    n_plots = len(funs)
    if n_plots > 1:
        assert n_plots == len(titles)
    else:
        if titles:
            print('Warning [plot]: will discard argument titles for a single plot')

    fig = plt.figure(figsize=(2.6+4.8*n_plots, 4.8))
    fig.suptitle(title, fontsize=14)

    for i in range(n_plots):
        vmin = np.min(vals[i])
        vmax = np.max(vals[i])
        cnorm = colors.Normalize(vmin=vmin, vmax=vmax)
        ax = fig.add_subplot(1, n_plots, i+1)
        ax.contourf(xx, yy, vals[i], 50, norm=cnorm, cmap='viridis')
        ax.axis('equal')
        fig.colorbar(cm.ScalarMappable(norm=cnorm, cmap='viridis'), ax=ax,  pad=0.05)
        ax.set_xlabel( r'$x$', rotation='horizontal' )
        ax.set_ylabel( r'$y$', rotation='horizontal' )
        if n_plots > 1:
            ax.set_title ( titles[i] )
    plt.show()

    if surface_plot:
        fig = plt.figure(figsize=(2.6+4.8*n_plots, 4.8))
        fig.suptitle(title+' -- surface', fontsize=14)

        for i in range(n_plots):
            vmin = np.min(vals[i])
            vmax = np.max(vals[i])
            cnorm = colors.Normalize(vmin=vmin, vmax=vmax)
            ax = fig.add_subplot(1, n_plots, i+1, projection='3d')
            ax.plot_surface(xx, yy, vals[i], norm=cnorm, cmap='viridis',
                        linewidth=0, antialiased=False)
            fig.colorbar(cm.ScalarMappable(norm=cnorm, cmap='viridis'), ax=ax,  pad=0.05)
            ax.set_xlabel( r'$x$', rotation='horizontal' )
            ax.set_ylabel( r'$y$', rotation='horizontal' )
            if n_plots > 1:
                ax.set_title ( titles[i] )
        plt.show()

class DirichletLinearOperator_2D(LinearOperator):
    """ A linear operator additionally applying homogeneous (essential) Dirichlet BCs before its evaluation. 

    Parameters
    ----------
    A : LinearOperator
    domain_space : continuous function space corresponding to the domain of A; must have kind property
    codomain_space : continuous function space corresponding to the codomain of A; must have kind property
    periodic : list of 2 booleans, periodicity in x and y directions; optional

    """
    def __init__(self, A, domain_space, codomain_space, periodic=[False, False]):
        
        self._A = A
        self._domain = A.domain
        self._codomain = A.codomain
        self._domain_space = domain_space
        self._codomain_space = codomain_space
        self._periodic = periodic
        self._BC = self._get_BCs()
        self._v_copy = self._domain.zeros()

    def _get_BCs(self):

        domain_space = self._domain_space
        periodic = self._periodic

        assert all([isinstance(P, bool) for P in periodic])
        if all([P == True for P in periodic]):
            return None
        
        u = element_of(domain_space, name='u')
        bcs = [EssentialBC(u, 0, side, position=0) for side in domain_space.domain.boundary]

        if domain_space.kind == H1Space:
            bcs_x = [bcs[0], bcs[1]]
            bcs_y = [bcs[2], bcs[3]]
        elif domain_space.kind == HcurlSpace:
            bcs_x = [bcs[2], bcs[3]]
            bcs_y = [bcs[0], bcs[1]]
        
        for P,bc in zip(periodic, (bcs_x, bcs_y)):
            if P:
                bc *= 0

        if domain_space.kind == H1Space:
            BC = bcs_x + bcs_y
        elif domain_space.kind == HcurlSpace:
            BC = [bcs_x, bcs_y]
        elif domain_space.kind == L2Space:
            BC = None
        else:
            raise ValueError('Domain Space kind not recognized')

        return BC

    @property
    def domain(self):
        return self._domain
    
    @property
    def codomain(self):
        return self._codomain
    
    @property
    def dtype(self):
        return self._A.dtype
    
    def tosparse(self):
        raise NotImplementedError
    
    def toarray(self):
        raise NotImplementedError
    
    def transpose(self, conjugate=False):
        return DirichletLinearOperator_2D(self._A.T, self._codomain_space, self._domain_space, self._periodic)
    
    def dot(self, v, out=None):
        """
        Projects v into the subspace of hom. DBCs of the domain function space and then applies A to the projected coefficient vector.
        Result is written to Vector out, if provided.
        
        """

        BC = self._BC
        vc = self._v_copy

        v.copy(out=vc)

        if BC is not None:
            if isinstance(self._A.domain, BlockVectorSpace): 
                for vi, bci in zip(vc, BC):
                    apply_essential_bc(vi, *bci)
            else:
                apply_essential_bc(vc, *BC)

        w = self._A.dot(vc, out=out)

        return w
