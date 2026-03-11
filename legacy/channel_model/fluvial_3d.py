import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from mpl_toolkits.mplot3d import Axes3D
import scipy.signal
import time
from matplotlib import animation
from scipy.interpolate import interp1d
import scipy.sparse
import random
from scipy.ndimage import gaussian_filter
import time
from numba import jitclass
from numba import jit
from genchannel import genchannel
from make_cutoff import make_cutoff

class fluvial:
    #facies: 0 shale; 1 pointbar/channel fill; 2 abandon channel; 3 levee;4 splay
    def __init__(self, station=1/32,complex_wid=500,b=80,Cf=0.0009,A=10.0, I=0.008,Q=0.9,sinu=1.5,dwratio=0.4,nlevel=10,pavul=0,nx=256,ny=128,nz=64,xmn=8,ymn=8,xsiz=16,ysiz=16,zsiz=3,rs=69069,ntg=1000,erode=0.1,lsplay=0,msplay=0,hsplay=0,ntime=10,azi0=0,bankratio=2,myidx=0):
        self.myidx=myidx
        self.b=b
        self.dwratio=dwratio
        self.LV_asym=0.9;
        self.lV_height=5*zsiz;
        dz=int((self.b*self.dwratio)/zsiz)
#        self.aggrad=[0,dz/12,dz/6,dz/6,dz/,dz/16]#aggradation rate in 3 stages
        
        ddd=7
        self.aggrad=[dz/ddd/np.random.uniform(3,4),dz/ddd/np.random.uniform(3,4),dz/ddd/np.random.uniform(4,5),dz/ddd/np.random.uniform(5,6),dz/ddd/np.random.uniform(5,6),dz/ddd/np.random.uniform(5,6),dz/ddd/np.random.uniform(7,8),dz/ddd/np.random.uniform(7,8),dz/ddd/np.random.uniform(7,8),dz/ddd/np.random.uniform(8,9),dz/ddd/np.random.uniform(9,10),dz/ddd/10,dz/ddd/10,dz/ddd/10]        
        
        
        
        self.bankratio=bankratio
        self.poro0=0.3
        self.poro=0.05*np.ones((nx,ny,nz))
        self.splay_dist=b*4;
        self.trunc_len=2*b
        self.A=A;
        
        self.Cf=Cf;
        self.I=I;
        self.Q=Q;
        self.azi0=0;
        self.sinu=sinu;
        
        self.nlevel=nlevel;
        self.pavul=pavul;
        self.nx=nx;
        self.ny=ny;
        self.nz=nz
        self.xmn=xmn;
        self.ymn=ymn;
        self.xsiz=xsiz;
        self.ysiz=ysiz;
        self.zsiz=zsiz
        self.rs=rs;
        self.ntg=ntg;
        self.erode=erode;
        self.lsplay=lsplay;
        self.msplay=msplay;
        self.hsplay=hsplay;
        self.ntime=ntime;
        g=9.8;
        self.g=g;
        self.us0=((g*Q*I)/(2.0*b*Cf))**(1.0/3.0);
        self.h0= Q/(2.0*b*self.us0);
        self.xmin=xmn-0.5*xsiz;
        self.ymin=ymn-0.5*ysiz;
        self.xmax=self.xmin+xsiz*nx;
        self.ymax=self.ymin+ysiz*ny;
        self.station_len=station*self.xmax
        self.x=np.linspace(xmn,self.xmax-xmn,self.nx);
        self.y=np.linspace(ymn,self.ymax-ymn,self.ny);
        
        self.step=(self.xsiz+self.ysiz)/2;
        self.step0=(self.xsiz+self.ysiz)*8;
        
        self.ndis0=int((((self.xmax-self.xmin)+(self.ymax-self.ymin))/2.0)/self.step)*4;
        self.ndis=self.ndis0;
        self.incr1=b/2;
        self.incr2=b/4;
        self.nthick=int(b/((xsiz+ysiz)/2.0));
        self.good=np.zeros([self.nx,self.ny])
        
        return
                    

    def generate_streamline(self,y0,x0=-1000,k=0.1,s=0.8,h=0.8,m=0):

        k=k*self.step0
        phi=np.arcsin(h);
        b1=2.0*np.exp(-k*h)*np.cos(k*np.cos(phi));
        b2=-1.0*np.exp(-2.0*k*h);
        mm0=s*np.random.normal(0,1,size=self.ndis0+40);
        mm0=mm0[20:-20]
        ar=np.array([1,b1,b2])
        theta=scipy.signal.lfilter([1],ar,mm0)+m
      
        self.cx0=np.cumsum(self.step0*np.cos(theta))
        self.cx0+=x0
        self.cy0=np.cumsum(self.step0*np.sin(theta))
        self.cy0+=y0;
        self.cx0=np.append([x0],self.cx0)
        self.cy0=np.append([y0],self.cy0)
        
#        self.cy=self.cy+np.sin(self.cx/self.xsiz/self.nx*np.pi)*self.ysiz*5
#        self.cx=np.arange(0,10000,4);
#        self.cy=self.ysiz*self.ny/2+np.sin(self.cx/16/10*np.pi)*16*10
        
        idx0=np.arange(self.cx0.size)
        if self.cx0.size<20:
            return 0
        else:
            self.ndis00=idx0[self.cx0>(self.xmax)][0]+4
            
        self.cx0=self.cx0[:self.ndis00]
        self.cy0=self.cy0[:self.ndis00]
        self.length=np.zeros(self.cx0.size)
        self.length[1:]=np.sqrt((self.cx0[1:]-self.cx0[:-1])**2+(self.cy0[1:]-self.cy0[:-1])**2)
        self.length=np.cumsum(self.length)
        # resample to length
        self.splx=UnivariateSpline(self.length,self.cx0,k=5,s=0)
        self.sply=UnivariateSpline(self.length,self.cy0,k=5,s=0)
        
        length=np.linspace(0,self.length[-1],self.cx0.size*18)
        self.step=length[1]-length[0]
        self.cx=self.splx(length)
        self.cy=self.sply(length)
        self.ndis=self.cx.size
        

        self.myinit=10
        #avoid centerline intercept boundary
        if (np.sum(self.cy>self.ymax-self.ny/2.5*self.ysiz*(1-self.chelev/self.nz/self.zsiz))+np.sum(self.cy<self.ymin+self.ny/2.5*self.ysiz*(1-self.chelev/self.nz/self.zsiz)))>0:
            return 0
        else:

            self.x0=self.cx[0]
            self.y0=self.cy[0]
            self.x1=self.cx[-1]
            self.y1=self.cy[-1]
            return 1
            
            

    
    
    
    
    def cal_curv(self,zzz=0):  

        dlength=np.sqrt((self.cx[1:]-self.cx[:-1])**2+(self.cy[1:]-self.cy[:-1])**2);
        dlength=np.append(0,dlength)
        self.dlength=dlength;
        length=np.cumsum(dlength);
#        s10=interp1d(length,self.cx, kind='linear')#N/S ratio=0.1
#        s20=interp1d(length,self.cy, kind='linear')#N/S ratio=0.1 
        s10=UnivariateSpline(length,self.cx,k=3,s=0)#N/S ratio=0.1 s=500
        s20=UnivariateSpline(length,self.cy,k=3,s=0)#N/S ratio=0.1   
        #transform coordinates to length   
        nstep=int((length[-1]-length[0])/self.step)+1
        self.length=np.linspace(length[0],length[-1]-0.1,nstep);
        
        self.cx=s10(self.length)
        self.cy=s20(self.length)
        #smooth centerline
        zz=0.2
        self.cx[1:-1]=self.cx[1:-1]*(1-zz)+zz/2*self.cx[:-2]+zz/2*self.cx[2:]
        self.cy[1:-1]=self.cy[1:-1]*(1-zz)+zz/2*self.cy[:-2]+zz/2*self.cy[2:]		
        
        s1=UnivariateSpline(self.length,self.cx,k=5,s=4000)#N/S ratio=0.1 s=500
        s2=UnivariateSpline(self.length,self.cy,k=5,s=4000)#N/S ratio=0.1 
      
        if zzz==0:
            self.cx=s10(self.length)
            self.cy=s20(self.length)        
        self.splx=s1
        self.sply=s2
        
        
        
        self.chwidth=self.b*np.ones(len(self.length))
        self.ndis=len(self.length)
        
        #calculate curvature
        vx=s1.derivative(n=1)
        ax=s1.derivative(n=2)
        vy=s2.derivative(n=1)
        ay=s2.derivative(n=2)
        curvature=(vx(self.length)*ay(self.length)-ax(self.length)*vy(self.length))/((vx(self.length)**2+vy(self.length)**2)**1.5);
        self.curv=curvature;    

        maxcurvr=max((self.curv))+0.0001;
        maxcurvl=max(-(self.curv))+0.0001;
        #calculate thalweg
        self.thalweg=self.curv.copy();
        self.thalweg[self.curv>=0]=0.5+self.curv[self.curv>=0]*0.25/maxcurvr;
        self.thalweg[self.curv<0]=0.5+self.curv[self.curv<0]*0.25/maxcurvl;
        #calculate dcsi/ds
        dcsids=(self.curv[1:]-self.curv[:-1])/(self.length[1:]-self.length[:-1]);
        self.dcsids=np.append(dcsids[0],dcsids);        
        #smooth these parameters
#        plt.figure(2)
#        plt.scatter(self.length,self.dcsids)
#        self.chwidth=self.tri_moving_average(10,self.chwidth)
#        self.thalweg=self.tri_moving_average(10,self.thalweg)
#        self.curv=self.tri_moving_average(10,self.curv)
#        self.dcsids=self.tri_moving_average(10,self.dcsids)
      
#        s5=UnivariateSpline(self.length,self.chwidth,w=weight,k=3,s=0)#N/S ratio=0.1
# =============================================================================
#         s4=UnivariateSpline(self.length,self.thalweg,k=5,s=0)#N/S ratio=0.1
# 
#         s8=UnivariateSpline(self.length,self.dcsids,k=5,s=0)#N/S ratio=0.1
# =============================================================================
        #resample at locations

        self.dlength=self.step;
        self.vx=vx(self.length);#save velocity
        self.vy=vy(self.length);
        
#
#        plt.plot(self.cx,self.cy)
       
#        self.cx[int(self.station_len/self.step):int(-self.station_len/self.step)]=s1(self.length[int(self.station_len/self.step):int(-self.station_len/self.step)])
#        self.cy[int(self.station_len/self.step):int(-self.station_len/self.step)]=s2(self.length[int(self.station_len/self.step):int(-self.station_len/self.step)])

#        self.chwidth=s5(self.length);
# =============================================================================
#         self.thalweg=s4(self.length);
#         self.dcsids=s8(self.length);
# =============================================================================
        

        return;
        
        
     
        
    
    def find_nearest(self,array,value):
        idx=(np.abs(array-value)).argmin();
        return idx
    

    
    def generatesplay(self,idis,dist):
        idis0=idis;
        for idis in np.arange(max(idis0-10,0),min(idis0+10,self.ndis)):
            #z is from bottom to top
            x1=self.cx[idis];
            y1=self.cy[idis];
            step=min(self.xsiz,self.ysiz)
            dist0=np.random.uniform(0.8*dist,1.2*dist)
            nst=int(dist0/step);       
            vx=self.vx[idis]
            vy=self.vy[idis]
            if self.curv[idis]>=0:#counter clock wise
                azi0=-np.pi/2;
            else:
                azi0=np.pi/2;
                
            c,s=np.cos(azi0),np.sin(azi0);
            R=np.array(((c,-s),(s,c)));            
            vx,vy=np.dot(R,np.array([vx,vy])); #change to breach direction
            
            #change breach point to bank
            x1=x1+vx*self.chwidth[idis];
            y1=y1+vy*self.chwidth[idis]; 
            ix=int((x1-self.xmin)/self.xsiz)
            iy=int((y1-self.ymin)/self.ysiz)            
            if x1>self.xmax or x1<0 or y1>self.ymax or y1<0 or self.facies[iy,ix]==1:
                return
            
            for ist in range(nst):
                azi=np.random.uniform(-np.pi/16,np.pi/16);
                #create rotation matrix
                c,s=np.cos(azi),np.sin(azi);
                R=np.array(((c,-s),(s,c)));             
                vx,vy=R.dot(np.array([vx,vy]));
                x2=x1+vx*step;
                y2=y1+vy*step;
                ix=int((x2-self.xmin)/self.xsiz)
                iy=int((y2-self.ymin)/self.ysiz)
                #bottom depth
                if x2>self.xmax or x2<0 or y2>self.ymax or y2<0 or (self.facies[iy,ix]==1).flatten().sum()>0:
                    break;

                self.facies[iy,ix]=4;
                self.poro[iy,ix]=(nst-ist)*self.poro0/4/nst+0.05
                self.shale[iy,ix]=ist*2/3/nst+1/3
                x1=x2;
                y1=y2;
    
        return;
    







        

        



    
    
    def simulation(self,nchannel=10):
#        fig = plt.figure(figsize=[18,3],dpi=100)
#        ax = plt.Axes(fig, [0., 0., 1., 1.])
#        ax.set_axis_off()
#        fig.add_axes(ax)
        
        self.itime=0;
        self.facies=np.zeros((self.nx,self.ny,self.nz));
        totalid=0
        self.totalid=totalid
        # generate center line for channel complex
        success=0

        self.out=0
        self.cz=int(self.dwratio*self.b/self.zsiz)
        self.chelev=(self.cz+1)*self.zsiz
   
        
        #initialize channel
        while(success==0):
            chy=np.random.uniform(self.ymin+(self.ymax-self.ymin)/2,self.ymin+(self.ymax-self.ymin)/2);
            angle=np.random.uniform(-np.pi/1800,np.pi/1800)
            success=self.generate_streamline(y0=chy,m=angle)
        idxx=np.ones(self.cx.size)
        self.chelev=(self.cz+1)*self.zsiz
        self.touch_b=0
        ntg=0
        self.myinit=10
        self.mybot=0
        
#        while(1):
        NNN=nchannel*10
        for ddd in range(NNN):
#            print(ddd)
            
            self.myinit-=1
#            print(str(self.cz)+': '+str(ntg))
#            if self.myinit<0:
#                ps=np.random.uniform(0,1)
#            
#            else:
#            #avulsion after at least 3 accretion                
#                ps=np.random.uniform(0,1)
#            #avulsion after touching complex boundary
#            self.touch_b=0
#            if self.touch_b==1:
#                ps=np.random.uniform(0,1)               

            


#            if ddd%8==7:
#                self.ps=0
#            else:
#                self.ps=1
            self.ps=1

#            current_step+=1
            self.out=0
            #exit when last channel is generated
#            if self.cz>=self.nz-2:
#                self.out=1
#                self.idxx=np.ones(self.idxx.shape)
#                self.cal_curv()
#                genchannel(self.b,self.xsiz,self.ysiz,self.chelev,self.zsiz,self.nx,self.ny,self.nz,self.cx,self.cy,self.x,self.y,self.vx,self.vy,self.curv,self.LV_asym,self.lV_height,self.ps,self.pavul,self.out,self.totalid,self.facies,self.poro,self.poro0,self.thalweg,self.chwidth,self.dwratio, [10000000000],NNN)
#                return 1;
            self.chelev=self.chelev+self.aggrad[int((self.totalid-1)/(NNN/len(self.aggrad)))]    
            #increase elevation when ntg is enough
#            if ntg>self.ntg:
#                if self.cz<=self.nz/8:
#                    self.out=1
#                    genchannel(self.b,self.xsiz,self.ysiz,self.chelev,self.zsiz,self.nx,self.ny,self.nz,self.cx,self.cy,self.x,self.y,self.vx,self.vy,self.curv,self.LV_asym,self.lV_height,self.ps,self.pavul,self.out,self.totalid,self.facies,self.poro,self.poro0,self.thalweg,self.chwidth,self.dwratio, [10000000000],NNN)                
#                    self.mybot=0+1
#                    self.cz=self.cz+self.aggrad[0]
#                    self.pavul=0.1
#                    self.chelev=(self.cz)*self.zsiz
#                    
#                elif self.cz<=(self.nz*9/16):
#                    self.out=1
#                    genchannel(self.b,self.xsiz,self.ysiz,self.chelev,self.zsiz,self.nx,self.ny,self.nz,self.cx,self.cy,self.x,self.y,self.vx,self.vy,self.curv,self.LV_asym,self.lV_height,self.ps,self.pavul,self.out,self.totalid,self.facies,self.poro,self.poro0,self.thalweg,self.chwidth,self.dwratio, [10000000000],NNN)                
#                        
#                    self.cz=self.cz+self.aggrad[1]
#                    self.pavul=0.1
#                    self.chelev=(self.cz)*self.zsiz
#                    self.mybot=int(self.nz/8)+1
#                else:
#                    self.out=1
#                    genchannel(self.b,self.xsiz,self.ysiz,self.chelev,self.zsiz,self.nx,self.ny,self.nz,self.cx,self.cy,self.x,self.y,self.vx,self.vy,self.curv,self.LV_asym,self.lV_height,self.ps,self.pavul,self.out,self.totalid,self.facies,self.poro,self.poro0,self.thalweg,self.chwidth,self.dwratio, [10000000000],NNN)                
#                    self.cz+=self.aggrad[2]
#                    self.pavul=0.1
#                    self.chelev=(self.cz)*self.zsiz
#                    self.mybot=int((self.nz*9/16))+1
                    
            totalid+=1;
            self.totalid=totalid
            
            #if avalsion happens this step
#            if self.ps<=self.pavul:  
            if self.touch_b==1: 
                #generate abandon channel fill
                self.out=1
                self.cal_curv()
                genchannel(self.b,self.xsiz,self.ysiz,self.chelev,self.zsiz,self.nx,self.ny,self.nz,self.cx,self.cy,self.x,self.y,self.vx,self.vy,self.curv,self.LV_asym,self.lV_height,self.ps,self.pavul,self.out,self.totalid,self.facies,self.poro,self.poro0,self.thalweg,self.chwidth,self.dwratio, [10000000000],NNN)                
                self.out=0
                
                success=0;
                while(success==0):
                    chy=np.random.uniform(self.ymin+(self.ymax-self.ymin)/2,self.ymin+(self.ymax-self.ymin)/2)
                    angle=np.random.uniform(-np.pi/1800,np.pi/1800)
                    success=self.generate_streamline(y0=chy,m=angle)
                    #generate channel width
                self.chwidth=self.b*np.ones(self.ndis);
                self.maxb=2*np.max(self.chwidth);
                #stop accretion if avulsion happens
                self.touch_b=0
                continue
            
        # if next step not going to perform avulsion generate point bar                
            #check if channel is too short
            if self.cx.size<20:
                return 0;
            #calculate curvature
            self.cal_curv()
            
            #calculate near bank velocity
            self.usbmat=np.zeros(self.ndis);
            for idis in np.arange(1,self.ndis):  
                dx=self.splx.integral(self.length[idis-1],self.length[idis])
                dy=self.sply.integral(self.length[idis-1],self.length[idis])
                dlength=np.sqrt(dx**2+dy**2)
                self.usbmat[idis]=self.b/(self.us0/dlength+2*(self.us0/self.h0)*self.Cf)*(-self.us0**2*self.dcsids[idis]+self.Cf*self.curv[idis]*(self.us0**4/self.g/self.h0**2+self.A*self.us0**2/self.h0)+self.us0/dlength*self.usbmat[idis-1]/self.b);
            self.usbmat=np.abs(self.usbmat)
            #standardize migration
            tmigrate=(self.xsiz+self.ysiz)*2.5
            vt=np.sqrt(self.vx**2+self.vy**2)
            
#            self.usbmat[:]=0.1
            damp=np.ones(self.cx.size)
            dist=np.arange(self.cx.size)
            damp=2-2/(1+np.exp(-np.abs(dist-dist.mean())/100))
#            self.usbmat_vx=np.sign(self.curv)*self.vy/vt*self.usbmat+0*self.us0*self.vx/vt
#            self.usbmat_vy=-np.sign(self.curv)*self.vx/vt*self.usbmat+0*self.us0*self.vy/vt
            self.usbmat_vx=np.sign(self.curv)*self.vy/vt*self.usbmat*damp+100*self.us0*self.vx/vt
            self.usbmat_vy=-np.sign(self.curv)*self.vx/vt*self.usbmat*damp+100*self.us0*self.vy/vt            
            self.usbmat_t=np.sqrt(self.usbmat_vx**2+self.usbmat_vy**2)
            self.maxmigrate=np.max(self.usbmat_t)
            self.E=tmigrate/self.maxmigrate#errorsion E
            #calculate migration
#            self.usbmat_vx[self.usbmat_t>2*tmigrate**2]=0
#            self.usbmat_vy[self.usbmat_t>2*tmigrate**2]=0
            self.cy[:21]=self.cy[16]
            self.cx[20:]=self.cx[20:]+self.usbmat_vx[20:]*self.E
            self.cy[20:]=self.cy[20:]+self.usbmat_vy[20:]*self.E
#            self.cy[-21:]=self.cy[-21]
#            self.cx[:-20]=self.cx[:-20]+self.usbmat_vx[:-20]*self.E
#            self.cy[:-20]=self.cy[:-20]+self.usbmat_vy[20:]*self.E            
#            for idx,idis in enumerate(np.arange(int(self.station_len/self.step),self.ndis-int(self.station_len/self.step))):
#                x1=self.cx[idis];
#                y1=self.cy[idis];
##                dist=self.usbmat[idis];
#                x2,y2=self.offset(x1,y1,self.usbmat_vx[idis]*self.E,self.usbmat_vy[idis]*self.E);
#                #damp migration at two end
#                rel_loc=1-abs(idx- )/(effect_dist)
#                damp_fac=(np.arctan((rel_loc-0.2)*10)/(np.pi/2)+1)/2
#
#                self.cx[idis]=self.cx[idis]+(x2-self.cx[idis])*damp_fac;
#                self.cy[idis]=self.cy[idis]+(y2-self.cy[idis])*damp_fac;  
# update grid near channel
#       create oxlake if two point is too close
            cut_dist=int(1*self.ndis/2);

            thresh=self.b*2.5;
            idxx=np.ones(self.ndis)
            make_cutoff(self.step,self.ndis,self.dlength,thresh,cut_dist,self.cx,self.cy,idxx,self.totalid)
        
            
#            center=self.complex_amp*np.sin(self.complex_w*self.cx+self.complex_dtheta)+self.ny*self.ysiz/2
#            ymax=center+self.complex_wid*self.complex_z[self.cz]
#            ymin=center-self.complex_wid*self.complex_z[self.cz]
            #perform avulsion if channel after avulsion touches complex boundary
#            if (np.sum(self.cy>self.ymax+5)+np.sum(self.cy<self.ymin-5))>0:
            if (np.sum(self.cy>self.ymax-self.ny/5*self.ysiz*(1-self.chelev/self.nz/self.zsiz))+np.sum(self.cy<self.ymin+self.ny/5*self.ysiz*(1-self.chelev/self.nz/self.zsiz)))>0:
                if self.totalid>10:
                    self.touch_b=1
                    
            #prevent two end from being neck cut off
#            nmin=np.argmin(self.cx)
#            nmax=np.argmax(self.cx)
#            if nmin!=0:
#                idxx[1:nmin]=0
#            if nmax!=self.cx.size-1:
#                idxx[nmax+1:-1]=0
            idxx[self.cx<self.x0]=0
            idxx[self.cx>self.x1]=0
            
#            self.cx[nmin]=0
#            if self.cx[nmax]<self.xmax+a.xsiz:
#                self.cx[nmax]=self.xmax+a.xsiz                
#            if self.cx[nmin]>a.xsiz or self.cx[nmax]<self.xmax-a.xsiz:
#                return 0; 
            
            self.out=0        
            #perform generation when change is large enough
#            print(np.sum((idxx==0)))
#            plt.plot(self.cx,self.cy)
            if self.totalid%10==9:
                mygood=(self.cx>self.xmin)*(self.cx<self.xmax)*(self.cy>self.ymin)*(self.cy<self.ymax)
                mygood=mygood.astype(bool)
                cx=self.cx[mygood]
                cy=self.cy[mygood]
                vx=self.vx[mygood]
                vy=self.vy[mygood]
                curv=self.curv[mygood]
                thalweg=self.thalweg[mygood]
                chwidth=self.chwidth[mygood]
                genchannel(self.b,self.xsiz,self.ysiz,self.chelev,self.zsiz,self.nx,self.ny,self.nz,cx,cy,self.x,self.y,vx,vy,curv,self.LV_asym,self.lV_height,self.ps,self.pavul,self.out,self.totalid,self.facies,self.poro,self.poro0,thalweg,chwidth,self.dwratio, [1000000000],NNN);

#                genchannel(self.b,self.xsiz,self.ysiz,self.chelev,self.zsiz,self.nx,self.ny,self.nz,self.cx,self.cy,self.x,self.y,self.vx,self.vy,self.curv,self.LV_asym,self.lV_height,self.ps,self.pavul,self.out,self.totalid,self.facies,self.poro,self.poro0,self.thalweg,self.chwidth,self.dwratio, [1000000000],NNN);
#                self.cy+=np.random.normal(0,4,self.cy.shape)
# =============================================================================
#             if np.sum(idxx)>4 or self.totalid%2==1:
#                 plt.scatter(self.cx[idxx.astype(bool)],self.cy[idxx.astype(bool)],color='blue',s=20,marker='s')
#                 plt.scatter(self.cx[(1-idxx).astype(bool)],self.cy[(1-idxx).astype(bool)],color='red',s=20,marker='s')
# #            myymin=self.cy[56:-56].min();myymax=self.cy[56:-56].max()
# =============================================================================
#            self.myymax=np.max((self.myymax,myymax+32))
#            self.myymin=np.min((self.myymin,myymin-32))
            self.cx=self.cx[idxx.astype(bool)];
            self.cy=self.cy[idxx.astype(bool)];
            if self.cx.size<20:
             return 0;			
            self.cx[0]=self.x0
            self.cy[0]=self.y0
            self.cx[-1]=self.x1
            self.cy[-1]=self.y1
            self.idxx=idxx;


        self.cal_curv()
#        plt.margins(0)
#        plt.scatter(self.cx,self.cy,color='yellow',s=20,marker='s')
#        plt.xlim([self.cx.min()+600,self.cx.max()-600])
#        plt.ylim([self.myymin,self.myymax])
#        fig.savefig('out'+str(self.myidx)+'.png', bbox_inches='tight', pad_inches=0,dpi=300)
#        plt.close()
 
    
    


                    



    
if __name__=='__main__':
# =============================================================================
    porosity=[]
    for i in np.arange(1):
         a=fluvial(myidx=i);
         a.simulation(nchannel=i%30)
         porosity.append(a.poro.copy())
         if i%100==99:
             # np.save('Porosity_multi'+str(i+100000)+'.npy',np.array(porosity))
             porosity=[]
#         plt.close()
         print(i)
#    plt.close()
#    for i in range(256):
#        plt.imshow(a.poro[i,:,:].T,vmax=0.38,vmin=0.05);plt.gca().invert_yaxis()
#        plt.savefig('./imgs/'+str(i+10000)+'.png',dpi=300);
#        plt.close()    
#    for i in range(64):
#        plt.imshow(a.poro[:,:,i].T,vmax=0.38,vmin=0.05);plt.gca().invert_yaxis()
#        plt.savefig('./imgs/'+str(i+10000)+'.png',dpi=300);
#        plt.close()
#                     
         

         
    # for i in range(32):
    #     plt.imshow(myface_final2[20,:,:,i]==6);
    #     plt.savefig('./imgs/'+str(i)+'.png')
    #     plt.close()   
    # import os
    # import imageio
    # images = []
    # filenames=os.listdir('./imgs')
    # for filename in filenames:
    #     images.append(imageio.imread('./imgs/'+filename))
    # imageio.mimsave('./added.gif', images)        
# =============================================================================
#              try:
#                  succeed=a.simulation();  
#              except:
#                  pass;
# #             for i in range(len(a.mycx)):
# #                 plt.plot(a.mycx[i],a.mycy[i])  
# #         plt.figure()    
# #         plt.imshow(a.facies[:,:,4])     
#          porosity.append(a.poro.copy())
# #         shale.append(a.shale.copy())
#          facies.append(a.facies.copy())
#          print(i) 
# #         plt.figure();plt.imshow(a.facies);plt.gca().invert_yaxis()
# #         plt.savefig(str(i)+'.png')
# #         plt.close()
#          plt.imshow(a.facies[:,:,3])
#          if i%100==99:
#              porosity=np.asarray(porosity)
# #             shale=np.asarray(shale)
#              facies=np.asarray(facies)
#              np.save('poro'+str(i)+'.npy',porosity)
# #             np.save('shale'+str(i)+'.npy',shale)
#              np.save('facies'+str(i)+'.npy',facies)    
# # 
# # #            np.save()
#              porosity=[]
#              facies=[]
#              shale=[]
#         
# =============================================================================
#             
# =============================================================================
    
#    plt.imshow(facies[0],cmap='gray',vmin=0,vmax=6)            
#    plt.figure()
#    plt.imshow(a.poro)
#    plt.figure()
#    plt.imshow(a.facies)