import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import multivariate_normal
from pathlib import Path
import json

class Layer:

    def __init__(self, nx, ny, nz, x_len, y_len, z_len, top_depth, dip, poro_ave, perm_ave, kzkx, active=None):
        """
        Initialize a Reservoir object with dimensions, grid properties, and well information.
        
        Parameters:
        - nx, ny, nz: Number of grid blocks in x, y, and z directions.
        - x_len, y_len, z_len: Physical dimensions of the reservoir in x, y, and z directions.
        - top_depth: Depth of the top of the reservoir.        
        - poro: Porosity value (can be scalar or array).
        - perm: Permeability value (can be scalar or array).
        - kzkx: Ratio of vertical to horizontal permeability.
        - active: Active flag grid (optional).
        """
        
        # Grid dimensions and size
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.x_len = x_len
        self.y_len = y_len
        self.z_len = z_len

        # Compute grid block dimensions
        self.dx = x_len / nx
        self.dy = y_len / ny
        self.dz = z_len / nz

        # Top depth and dip angle of the reservoir
        self.top_depth = top_depth
        self.dip = dip

        # Initialize lists to store well locations, types, names, and grid coordinates
        self.well_locs = []
        self.well_types = []
        self.well_names = []
        self.well_grids = []

        # Create grid for reservoir surface
        self.x = np.linspace(0, self.x_len, self.nx + 1)
        self.y = np.linspace(0, self.y_len, self.ny + 1)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
        
        # Depth information
        self.z1 = self.Y*np.tan(self.dip/180*np.pi) + self.top_depth
        self.z2 = self.z1 + self.z_len
        self.zz = [self.z1, self.z2]
        
        # Average porosity and permeability
        self.poro_ave = poro_ave
        self.perm_ave = perm_ave

        # If porosity and permeability are scalar, create uniform property arrays
        if isinstance(self.poro_ave, (float, int)):
            self.poro_mat = self.poro_ave * np.ones((self.nx, self.ny, self.nz))
        if isinstance(self.perm_ave, (float, int)):
            self.perm_mat = self.perm_ave * np.ones((self.nx, self.ny, self.nz))

        self.kzkx = kzkx

        # If active is not provided, assume all grids are active
        if active is None:
            self.active = np.ones((self.nx, self.ny, self.nz))

class GaussianLayer(Layer):
    pass

class LobeLayer(Layer):
    
    def __init__(self, nx, ny, nz, x_len, y_len, z_len, top_depth, dip, poro_ave, poro_std, perm_ave, perm_std, kzkx, ntg, active=None):

        super().__init__(nx, ny, nz, x_len, y_len, z_len, top_depth, dip, poro_ave, perm_ave, kzkx, active)

        self.poro_std = poro_std
        self.perm_std = perm_std
        self.ntg = ntg
        self.detailed_lobe_data = []


    def create_geology(self, dhmin=4, dhmax=4, rmin=42, rmax=44, asp=1.5, theta0=0, m=100, upthinning = True, bouma_factor = 0):

        # np.random.seed(20)
        allfacies, allporo, self.allsurface = self.lobemodeling(dhmax=dhmax, dhmin=dhmin, rmin=rmin, rmax=rmax, asp=asp,
                                                                theta0=theta0, m=m, upthinning=upthinning, bouma_factor=bouma_factor)     
        self.lobe_poro = allporo[-1]
        
        # np.random.seed(250)

        sand_filt = [1.5,2.5,1.5]
        facies_filt = [2.5,5,2.5]
        sand_nug = 0.05
        
        # change axis from zyx to xyz
        self.lobe_poro = np.swapaxes(self.lobe_poro, 0, -1)

        # facies modeling
        # these are perturbation
        lambda_perturb = 0.1 # degree of local perturbation
        self.active = gaussian_filter(np.random.normal(0,1,(3*self.nx, 3*self.ny, 3*self.nz)), facies_filt, mode='wrap')
        self.active = self.active[self.nx:2*self.nx, self.ny:2*self.ny, self.nz:2*self.nz]
        self.active = self.lobe_poro + lambda_perturb*self.active
        self.active = self.active > np.percentile(self.active.flatten(), (1-self.ntg)*100)

        # property modeling
        self.poro_mat = np.random.normal(0, 1, (3*self.nx, 3*self.ny, 3*self.nz))
        self.poro_nug = np.random.normal(0, 1, (3*self.nx, 3*self.ny, 3*self.nz))
        self.poro_mat = gaussian_filter(self.poro_mat, sand_filt, mode='wrap') + sand_nug*self.poro_nug
        self.poro_mat = self.poro_mat[self.nx:2*self.nx, self.ny:2*self.ny, self.nz:2*self.nz]
        self.poro_mat = self.lobe_poro + lambda_perturb*self.poro_mat
        self.perm_mat = self.poro_mat.copy()

        # for uppepr unit
        # percentile transform
        mean = [0,0]
        var = [[1, 0.6], [0.6, 1]]
        mm = multivariate_normal(mean,var)
        sand1 = mm.rvs(int(self.poro_mat.size*1.2))
        sand1[:,0] = sand1[:,0]*self.poro_std + self.poro_ave
        sand1[:,1] = sand1[:,1]*self.perm_std + self.perm_ave

        sand1 = sand1[sand1[:,0]<(self.poro_ave+5*self.poro_std)]
        sand1 = sand1[sand1[:,0]>(self.poro_ave-5*self.poro_std)]
        sand1 = sand1[sand1[:,1]<(self.perm_ave+5*self.perm_std)]
        sand1 = sand1[:self.poro_mat.size]

        # quantile transformation to force it to be gaussian
        poro_flattened = self.poro_mat.flatten()
        poro_flat_order = np.argsort(poro_flattened)
        sand1 = sand1[np.argsort(sand1[:self.poro_mat.size,0])]
        poro_flattened[poro_flat_order] = sand1[:,0]
        perm_flattened = poro_flattened.copy()
        perm_flattened[poro_flat_order] = sand1[:,1]
        self.poro_mat = poro_flattened.reshape(self.poro_mat.shape)
        self.perm_mat = perm_flattened.reshape(self.poro_mat.shape)

        self.active = self.active.astype(int)
        self.poro_mat = self.poro_mat*self.active
        self.perm_mat = (10**self.perm_mat)*self.active

            # Store facies map for later use (swap axes to match xyz convention)
        if allfacies and len(allfacies) > 0:
            self.facies = np.swapaxes(allfacies[-1].copy(), 0, -1).astype(int)
        else:
            self.facies = None

        # Optionally select actual geological lobes and boost their porosity as "sweet" lobes
        if n_sweet and n_sweet > 0:
            self.select_and_boost_sweet_lobes(n=n_sweet, amp_min=sweet_amp_min, amp_max=sweet_amp_max)


    def lobemodeling(self, dhmax=4, dhmin=4, rmin=42, rmax=44, asp=1.5, theta0=0, m=100, upthinning = True, bouma_factor = 0):

        facies = np.zeros((self.nz, self.ny, self.nx))
        poro = facies.copy() - 0.1
        allsurface = []
        surface = 0.000001*np.ones((self.ny, self.nx))
        surface0 = surface.copy()
        lat_size = self.nx * self.ny
        loc_idx = np.arange(lat_size)
        theta0 = theta0 / 180 * np.pi
        allfacies = []
        allporo = []
        allsurface.append(surface.copy())
        start = 0  # when to start add lobe
        
        # dv = []
        i = 0
        iiii = 10000
        while(i < iiii-1):
            
            surface0 = surface.copy()
            #calculate prob
            zz = surface
            prob = (1/(surface-zz.min()+0.001)**m) / np.sum(1/(surface-zz.min()+0.001))
            
            #choose loc
            prob = prob / np.sum(prob)
            prob_flat = prob.flatten()
            loc = np.random.choice(loc_idx, p=prob_flat)
            y = loc // self.nx
            x = loc - self.nx*y
        
            theta = theta0 + np.random.normal(0, 20/180*np.pi)
            dh = np.random.uniform(dhmin, dhmax)
            r = np.random.uniform(rmin, rmax)
            
            # # This is all new block that should be removed and 3 lines above should be uncommented out
            # x_norm = x/self.nx
            # y_norm = y/self.ny
            # theta = theta0 - 140/180*np.pi*(x_norm-0.5)
            # asp = 2*y_norm + 1
            # dh = 20*y_norm + 15
            # # r = 12*y_norm + 10
            # r = -12*y_norm + 22

            # update surface
            self.update_surface(x, y, r, asp, theta, dh, surface)
            if i!=0:          
                # do healing correction
                surface2 = surface.copy()  # initial volume
                surface = surface0 + (surface-surface0)*(1-(surface0/surface0.max())**1.2)  # after healing volume
                dsurface = (surface-surface0)
                
                # compensate volume
                surface = surface0 + dsurface * (np.sum(surface2-surface0)/np.sum(dsurface+0.000000001))
                
            # assign property
            dz = surface-surface0
            ychange,xchange = np.where(dz>0)
            if i > start-0.1:
                self.detailed_lobe_data.append([x,y,dh,r,asp,theta])
                allsurface.append(surface.copy())
                self.assign_prop(xchange, ychange, x, y, theta, surface0, surface, facies, r, poro, dz, asp, i-start+1, upthinning, bouma_factor) 
            i+=1
            if i==iiii or surface.max()>=self.nz+4:
                allfacies.append(facies.copy())
                allporo.append(poro.copy())
                break
        return allfacies, allporo, allsurface

    def select_and_boost_sweet_lobes(self, n=5, amp_min=0.10, amp_max=0.15):
        """
        Randomly select n geological lobes and increase their porosity to make them "sweet".
        
        This method:
        1. Identifies all unique lobe IDs in the facies array (excluding 0, which is background)
        2. Randomly selects n lobes
        3. Boosts porosity within each selected lobe by a random amount
        4. Records metadata including lobe ID, voxel locations, and boost amount
        
        Parameters
        - n: number of lobes to select and boost (int)
        - amp_min, amp_max: porosity increase range (float)
        
        Effects
        - updates self.poro_mat in-place (adds amplitude to all cells in selected lobes)
        - creates self.sweet_mask (bool array marking boosted cells)
        - creates self.sweet_metadata (list of dicts with lobe_id, boost amount, voxel_coords)
        
        Requires
        - self.facies to be populated (run create_geology() first)
        - self.poro_mat to exist
        """
        if not hasattr(self, 'facies') or self.facies is None:
            raise AttributeError('facies map not found; run create_geology() first')
        if not hasattr(self, 'poro_mat'):
            raise AttributeError('poro_mat not found; run create_geology() first')
        
        # Find all unique lobe IDs (excluding 0 = background/non-sand)
        lobe_ids = np.unique(self.facies)
        lobe_ids = lobe_ids[lobe_ids > 0]  # keep only active lobe IDs
        
        if len(lobe_ids) < n:
            raise ValueError(f'Only {len(lobe_ids)} lobes available, but requested to boost {n}')
        
        # Randomly select n lobes
        selected_lobes = np.random.choice(lobe_ids, size=int(n), replace=False)
        
        # Initialize tracking structures
        self.sweet_mask = np.zeros_like(self.poro_mat, dtype=bool)
        self.sweet_metadata = []
        
        # Boost porosity in each selected lobe
        for lobe_id in selected_lobes:
            # Create mask for this lobe
            lobe_mask = (self.facies == lobe_id)
            
            # Random sample boost amplitude
            amp = float(np.random.uniform(amp_min, amp_max))
            
            # Apply boost to porosity
            self.poro_mat[lobe_mask] += amp
            
            # Update sweet mask
            self.sweet_mask = self.sweet_mask | lobe_mask
            
            # Get voxel coordinates (i, j, k) for all cells in this lobe
            voxel_coords = np.argwhere(lobe_mask)  # Returns (nx, ny, nz) indices as Nx3 array
            
            # Record metadata
            metadata = {
                'lobe_id': int(lobe_id),
                'boost_amount': float(amp),
                'cell_count': int(np.sum(lobe_mask)),
                'voxel_coords': voxel_coords.tolist()  # Convert to list for JSON serialization
            }
            self.sweet_metadata.append(metadata)
        
        # Clamp porosity to valid range [0, 0.9]
        self.poro_mat = np.clip(self.poro_mat, 0.0, 0.9)

    def update_surface(self, x, y, r, asp, theta, dh, surface):
        for ii in np.arange(max(0,int(y-r*asp)), min(int(y+r*asp), self.ny)):
            for jj in np.arange(max(0,int(x-r*asp)), min(int(x+r*asp), self.nx)):
                dx = jj - x
                dy = ii - y
                # rotated coordinate
                dx2 = dx*np.cos(theta) - dy*np.sin(theta)
                dy2 = dx*np.sin(theta) + dy*np.cos(theta)
                r1 = np.sqrt((dx2/asp)**2 + dy2**2)

                if r1**2 <= r**2:
                    dz0 = -dh/(r**2)*(r1**2) + dh
                    surface[ii,jj] = surface[ii,jj] + dz0    
        return 0


    def assign_prop(self, xchange, ychange, x, y, theta, surface0, surface, facies, r, poro, dz, asp, i, upthinning, bouma_factor):
        for n in range(xchange.size):
            ii = ychange[n]
            jj = xchange[n]    
            dx = jj - x
            dy = ii - y
            # rotated coordinate
            dx2 = dx*np.cos(theta) - dy*np.sin(theta)
            dy2 = dx*np.sin(theta) + dy*np.cos(theta)
            r1 = np.sqrt((dx2/asp)**2 + dy2**2)            
            # assign value
            bot = int(np.rint(surface0[ii,jj]))
            top = int(min(np.rint(surface[ii,jj]), self.nz))
            # facies[max(top-4,0):top,ii,jj][facies[max(top-4,0):top,ii,jj]==0]=i
            poromin = 0.05
            poromax = 0.3 * ((1-bot/self.nz)/2+0.5) + 0.05 if upthinning else 0.35
            if top > bot:
                facies[bot:top,ii,jj][facies[bot:top,ii,jj]==0] = i
                # if top-bot>=1:
                for kk in np.arange(bot, top):
                    # Rz=(1-np.sqrt((kk-bot)/dz[ii,jj]))*r
                    upthinning_factor = ((1-kk/self.nz)/2+0.5) if upthinning else 1
                    poro[kk,ii,jj] = 0.3 * ((top-kk)/(top-bot)) * (1-(r1/r)) * upthinning_factor + 0.05
                    poronorm = (poro[kk,ii,jj] - poromin) / (poromax - poromin)
                    bouma_seq_lims = [0,0.1,0.2,0.3,0.4,1]
                    for bouma_idx in range(len(bouma_seq_lims)-1):
                        if bouma_seq_lims[bouma_idx] <= poronorm < bouma_seq_lims[bouma_idx+1]:
                            bouma_seq_mid = (bouma_seq_lims[bouma_idx] + bouma_seq_lims[bouma_idx+1]) / 2
                            poronorm = (1-bouma_factor) * (poronorm - bouma_seq_mid) + bouma_seq_mid
                            break
                    # if 0 <= poronorm < 0.4:
                    #     poronorm = (1-bouma_factor) * (poronorm - 0.2) + 0.2
                    # elif poronorm < 0.5:
                    #     poronorm = (1-bouma_factor) * (poronorm - 0.45) + 0.45
                    # elif poronorm < 0.6:
                    #     poronorm = (1-bouma_factor) * (poronorm - 0.55) + 0.55
                    # elif poronorm < 0.7:
                    #     poronorm = (1-bouma_factor) * (poronorm - 0.65) + 0.65
                    # elif poronorm <= 1:
                    #     poronorm = (1-bouma_factor) * (poronorm - 0.85) + 0.85
                    # else:
                    #     print("poronorm is", poronorm, "that's weird")
                    poro[kk,ii,jj] = poromin + poronorm * (poromax - poromin)

                # if facies[kk,ii,jj]==i:
                # poro[kk,ii,jj]=0.05 
        return 0
    

    def set_property(self, prop_name, prop_value):
        if prop_name == 'porosity' or prop_name == 'por' or prop_name == 'poro' or prop_name == 'phi':
            self.poro_ave = prop_value
        elif prop_name == 'permeability' or prop_name == 'perm':
            self.perm_ave = prop_value
        elif prop_name == 'thickness' or prop_name == 'z_len':
            self.z_len = prop_value
        elif prop_name == 'net-to-gross':
            self.ntg = prop_value
        elif prop_name == 'x_length' or prop_name == 'x_len':
            self.x_len = prop_value
        elif prop_name == 'y_length' or prop_name == 'y_len':
            self.y_len = prop_value

    def export_sweet_metadata(self, dirpath, prefix='sweet', save_mask=True, save_metadata=True):
        """
        Export sweet-lobe mask and metadata to disk.

        Parameters
        - dirpath: directory path (str or Path) where files will be written
        - prefix: filename prefix
        - save_mask: if True, saves mask as <prefix>_mask.npy
        - save_metadata: if True, saves metadata as <prefix>_metadata.json

        Returns a dict with paths written.
        """
        if not hasattr(self, 'sweet_mask') or not hasattr(self, 'sweet_metadata'):
            raise AttributeError('No sweet metadata found. Run inject_sweet_lobes() first.')

        out = {}
        p = Path(dirpath)
        p.mkdir(parents=True, exist_ok=True)

        if save_mask:
            mask_path = p / f"{prefix}_mask.npy"
            np.save(str(mask_path), self.sweet_mask)
            out['mask'] = str(mask_path)

        if save_metadata:
            meta_path = p / f"{prefix}_metadata.json"
            # Convert any non-JSON native types (tuples) to lists inside metadata
            serializable = []
            for m in self.sweet_metadata:
                mm = m.copy()
                # convert tuples to lists for JSON safety
                if 'center_idx' in mm and isinstance(mm['center_idx'], tuple):
                    mm['center_idx'] = list(mm['center_idx'])
                if 'bbox' in mm and isinstance(mm['bbox'], tuple):
                    mm['bbox'] = list(mm['bbox'])
                serializable.append(mm)
            with open(str(meta_path), 'w') as fh:
                json.dump(serializable, fh, indent=2)
            out['metadata'] = str(meta_path)

        return out
