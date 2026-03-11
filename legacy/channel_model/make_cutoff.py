import numpy as np
from numba import jit
@jit(nopython=True)
def make_cutoff(step,ndis,dlength,thresh,cut_dist,cx,cy,idxx,totalid):
    for i in np.arange(int(thresh/step),ndis-2,2):
        for j in np.arange(min(i+int(1.5*thresh/step),ndis),min(i+int(0.1*ndis),ndis),2):
            if np.sqrt((cx[i]-cx[j])**2+(cy[i]-cy[j])**2)<thresh:
#                            idxx[max(i-3,0):min(j+3,self.ndis)]=0;     
                idxx[max(i+1,0):min(max(j-1,0),ndis)]=0;