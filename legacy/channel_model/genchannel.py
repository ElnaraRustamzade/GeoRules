import numpy as np
from numba import jit
@jit(nopython=True)
def find_near_grid(cx,cy,good,xsiz,ysiz,b,nx,ny):
    nd=int(6*b/min(xsiz,ysiz))
    for i in range(cx.size):
        mynx=int(cx[i]/xsiz)
        myny=int(cy[i]/ysiz)
        for j in range(max(0,mynx-nd),min(nx,mynx+nd)):
            for k in range(max(0,myny-nd),min(ny,myny+nd)):
                good[j,k]=1
    return 0

@jit(nopython=True)
def mychannel(nz,lV_height,LV_asym,curv,mynx,myny,localx,localy,x,y,vx,vy,cy,cx,thalweg,chelev,idz,out,ps,zsiz,dd,totalid,
              cutoff,pavul, poro0, facies, poro, chwidth,idmat,nx,ny,dwratio,NN):
    #use cross product to determine location of the grid relative to streamline coordinate                                    
#    dxy=np.array([vx[idis],vy[idis]]);
    for myid in range(localx.size):
        idx=mynx[myid]
        idy=myny[myid]              
        dist=idmat[dd[myid],myid]
        idis=dd[myid];#corresponding channel location
        if dist>chwidth[0]:
            continue;
#                if(idmat[dd[idt],idt]<self.b): 
  

# =============================================================================
#         elif dist<=2*chwidth[idis] and dist>1*chwidth[idis]:
#             LV_wid=1*chwidth[idis] #levee width 2* channel width
#             idis=dd[myid]
#             lvdepth=5*zsiz;
#             #check if the levee is at point bar side          
#             dx2=x[idx]-cx[idis];dy2=y[idy]-cy[idis];
#             indicator=dx2*vy[idis]-dy2*vx[idis];    
#             curvmax=max(curv.max(),-curv.max())
#             
#             if curv[idis]*indicator>=0: #point is at cut bank side
#                 F=1+LV_asym*abs(curv[idis])/curvmax;
#             else:
#                 F=1-LV_asym*abs(curv[idis])/curvmax;
#             dist=dist-chwidth[idis];    
#             lvtop=lV_height*(dist/LV_wid/F)*np.exp(-dist/LV_wid/F)+chelev;  
#             lvtop=max(lvtop,chelev);
#             lbot=chelev-lvdepth*min((LV_wid*F-dist)/(LV_wid*F),1);
#             
#             poro[idx,idy,max(int(lbot/zsiz),0):min(int(lvtop/zsiz)+1,nz)][facies[idx,idy,max(int(lbot/zsiz),0):min(int(lvtop/zsiz)+1,nz)]==0]=(1-dist/LV_wid)*0.1+0.1*poro0
#             facies[idx,idy,max(int(lbot/zsiz),0):min(int(lvtop/zsiz)+1,nz)][facies[idx,idy,max(int(lbot/zsiz),0):min(int(lvtop/zsiz)+1,nz)]==0]=1;                        
#             
# #            facies[idx,idy,max(int(lbot/zsiz),0):min(int(lvtop/zsiz)+1,nz)]=1;                        
# #            poro[idx,idy,max(int(lbot/zsiz),0):min(int(lvtop/zsiz)+1,nz)]=min(max(1-dist/LV_wid,1),0)*0.1+0.3*poro0;
# 
# =============================================================================
        else:  
#                idis=dd[idt];
            t=dwratio*chwidth[idis];
            WW=chwidth[idis]*2;                 
            dx2=x[idx]-cx[idis];dy2=y[idy]-cy[idis];
        #    dxy2=np.array([dx2,dy2]);
            indicator=dx2*vy[idis]-dy2*vx[idis];
            if indicator>0:    #on the right
                wid=dist+chwidth[idis];
                wid=max(wid,0)
            else:
                wid=chwidth[idis]-dist; #on the left
                wid=max(wid,0)
            a=min(0.99999,max(0.00001,thalweg[idis]));
        #    print(a)
            if(a<0.5):
                by=-np.log(2.0)/np.log(a);
                chbot = chelev - 4.0*t*(wid/WW)**by*(1.0-(wid/WW)**by)
                #calculate maximum depth
                wid2=a*WW
                maxD= 4.0*t*(wid2/WW)**by*(1.0-(wid2/WW)**by)
        
            else:
                ddy=-np.log(2)/np.log(1-a);
                chbot = chelev - 4.0*t*(max(0,1.0-wid/WW))**ddy*(1.0-(max(0,1.0-wid/WW))**ddy)
                #calculate maximum depth
                wid2=a*WW
                maxD = 4.0*t*(max(0,1.0-wid2/WW))**ddy*(1.0-(max(0,1.0-wid2/WW))**ddy)      
            
            ddz=int(max(((chelev-chbot)/zsiz),0));
#            if (dd[myid] in cutoff): 
                #generate point bar
            facies[idx,idy,idz-ddz:idz]=1;
            maxDint=int(maxD/zsiz)
            for myz in np.arange(ddz):
                poro[idx,idy,idz-myz]=(0.9/(1+np.exp(-4*(myz/(maxDint)-0.2)))+0.1)*(poro0)*(min((chelev-chbot)/(maxD),1)**2)+0.1;


#*
# =============================================================================
#                 if ddz<t//zsiz-1:
#                     poro[idx,idy,idz-ddz:min(nz,idz-ddz+int((1-ddz/(t/zsiz))*2))]=0.05
# =============================================================================
#                print(dist/WW)
# =============================================================================
#             else:
#                 #oxlake mud plug
#                 facies[idx,idy,idz-ddz:idz]=3;
#                 poro[idx,idy,idz-ddz:idz]=np.arange(ddz,0,-1)/(ddz)*(0.08*1);
# =============================================================================
#                mydd=1

                
#                mydd=1
            
#            if mydd==1:
#                LV_wid=1*chwidth[idis] #levee width 2* channel width
#                idis=dd[myid]
#                lvdepth=5*zsiz;
#                #check if the levee is at point bar side          
#                dx2=x[idx]-cx[idis];dy2=y[idy]-cy[idis];
#                indicator=dx2*vy[idis]-dy2*vx[idis];    
#                curvmax=max(curv.max(),-curv.max())
#                
#                if curv[idis]*indicator>=0: #point is at cut bank side
#                    F=1+LV_asym*abs(curv[idis])/curvmax;
#                else:
#                    F=1-LV_asym*abs(curv[idis])/curvmax;
#                dist=dist-chwidth[idis];    
#                lvtop=lV_height*(dist/LV_wid/F)*np.exp(-dist/LV_wid/F)+chelev;  
#                lvtop=max(lvtop,max(chbot,0));
#                lbot=chelev-lvdepth*min((LV_wid*F-dist)/(LV_wid*F),1);
#                poro[idx,idy,max(int(lbot/zsiz),0):min(int(lvtop/zsiz)+1,nz)][facies[idx,idy,max(int(lbot/zsiz),0):min(int(lvtop/zsiz)+1,nz)]==0]=(1-dist/LV_wid)*0.1+0.1*poro0;
#                facies[idx,idy,max(int(lbot/zsiz),0):min(int(lvtop/zsiz)+1,nz)][facies[idx,idy,max(int(lbot/zsiz),0):min(int(lvtop/zsiz)+1,nz)]==0]=1;                        
                
#                facies[idx,idy,max(int(lbot/zsiz),0):idz-ddz+1]=1;                        
#                poro[idx,idy,max(int(lbot/zsiz),0):idz-ddz+1]=min(max(1-dist/LV_wid,1),0)*0.1+0.3*poro0;

#



    return 0;        



#@jit(nopython=True)
def genchannel(b,xsiz,ysiz,chelev,zsiz,nx,ny,nz,cx,cy,x,y,vx,vy,curv,LV_asym,lV_height,ps,pavul,out,totalid,facies,poro,poro0,thalweg,chwidth,dwratio,cutoff,NN=800):
    #SEARCH GRID NEAR CONTROL POINT AND SEE IF THEY ARE INSIDE THE CHANNEL
    idz=int(chelev/zsiz)
    ndis=cx.size
    good=np.zeros((nx,ny))
    find_near_grid(cx,cy,good,xsiz,ysiz,b,nx,ny)    
    mynx,myny=np.where(good==1)
    cx2=cx.reshape(ndis,1)
    cy2=cy.reshape(ndis,1)
    localx=x[mynx]
    localy=y[myny]    
    idmat=np.sqrt((cx2-localx)**2+(cy2-localy)**2);            
    dd=idmat.argmin(axis=0)#minimum distance for each grid
    

    mychannel(nz,lV_height,LV_asym,curv,mynx,myny,localx,localy,x,y,vx,vy,cy,cx,thalweg,chelev,idz,out,ps,zsiz,dd,totalid,
              cutoff,pavul, poro0, facies, poro, chwidth,idmat,nx,ny,dwratio,NN)
                    

#        self.facies[self.idx,self.idy,:]=10;
    
    return 0; 
#genchannel(a.b,a.xsiz,a.ysiz,a.chelev,a.zsiz,a.nx,a.ny,a.nz,a.cx,a.cy,a.x,a.y,a.vx,a.vy,a.curv,a.LV_asym,a.lV_height,a.ps,a.pavul,a.out,a.totalid,a.facies,a.poro,a.poro0,a.thalweg,a.chwidth,a.dwratio,[1,2,3,4,5,6,7])
#lp = LineProfiler()
#lp_wrapper = lp(genchannel)
#lp_wrapper(a.b,a.xsiz,a.ysiz,a.chelev,a.zsiz,a.nx,a.ny,a.nz,a.cx,a.cy,a.x,a.y,a.vx,a.vy,a.curv,a.LV_asym,a.lV_height,a.ps,a.pavul,a.out,a.totalid,a.facies,a.poro,a.poro0,a.thalweg,a.chwidth,a.dwratio,[1,2,3,4,5,6,7])
#lp.print_stats()
#genchannel(a.b,a.xsiz,a.ysiz,a.chelev,a.zsiz,a.nx,a.ny,a.nz,a.cx,a.cy,a.x,a.y,a.vx,a.vy,a.curv,a.LV_asym,a.lV_height,a.ps,a.pavul,a.out,a.totalid,a.facies,a.poro,a.poro0,a.thalweg,a.chwidth,a.dwratio, [1000000])    
#self.b,self.xsiz,self.ysiz,self.chelev,self.zsiz,self.nx,self.ny,self.nz,self.cx,self.cy,self.x,self.y,self.vx,self.vy,self.curv,self.LV_asym,self.lV_height,self.ps,self.pavul,self.out,self.totalid,self.facies,self.poro,self.poro0,self.thalweg,self.chwidth,self.dwratio,    