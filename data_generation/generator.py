import numpy as np
import gstools as gs

def add_boundary_layers(modelin, nx, ny, nxin, nyin):
    nxout, nyout = nx - nxin, ny - nyin
    data = np.zeros((nx, ny))
    data[nxout//2:-nxout//2, :-nyout] = modelin
    # add boundary layers
    data[:nxout//2, :] = data[nxout//2, :]
    data[-nxout//2:, :] = data[-nxout//2-1, :]
    data[:, -nyout:] = np.expand_dims(data[:, -nyout-1], axis=1)
    return data

def generate_vs(nxin, nyin, vsbg):
    rf = generate_2DRF(nxin, nyin, len_scale=[64, 8], sigma=20, nu=1.5, seed=-1)
    vs = perturbation(vsbg, rf)
    return vs

def generate_vp(nxin, nyin, vs):
    vp = brocher_vp_from_vs(vs)
    rf = generate_2DRF(nxin, nyin, len_scale=[64, 8], sigma=2, nu=2.5, seed=-1)
    vp = perturbation(vp, rf)
    return vp

def generate_2DRF(nx, ny, len_scale=32, sigma=10, nu=1.5, seed=-1):
    '''
    Generate random fields with same correlation length scale in x and y
    Check https://geostat-framework.readthedocs.io/projects/gstools/en/stable/ for more details
    n - the shape of the output array
    ncorr - correlation length in the unit of grids
    seed - if needed
    ''' 
    x, y = range(nx), range(ny)
    model = gs.Matern(dim=2, var=sigma**2, len_scale=len_scale, nu=nu)
    if seed >= 0:
        srf = gs.SRF(model, seed=seed)
    else:
        srf = gs.SRF(model)

    dV = srf((x, y), mesh_type='structured')
    return dV

def perturbation(bg, rf):
    '''
    bg: background model, (nx, ny)
    rf: random field, (nx, ny)
    '''
    out = bg * (1 + (rf / 100))
    return out

def brocher_vp_from_vs(vs):
    '''
    Calculate vp from vs using Brocher, 2005
    valid for vs between 0 and 4.5 km/s
    vs: m/s
    vp: m/s
    '''
    vs = vs / 1000  # to km/s
    vp = 0.9409 + 2.0947*vs - 0.8206*vs**2 + 0.2683*vs**3 - 0.0251*vs**4
    vp = vp * 1000  # to m/s
    return vp

def brocher_rho_from_vp(vp):
    '''
    Calculate rho from vp using Brocher, 2005
    valid for vp between 1.5 and 8.5 km/s
    vp: m/s
    rho: kg/m^3
    '''
    vp = vp / 1000  # to km/s
    rho = 1.6612*vp - 0.4721*vp**2 + 0.0671*vp**3 - 0.0043*vp**4 + 0.000106*vp**5  # g/cc
    rho = rho * 1000  # to kg/m^3
    return rho