import numpy as np
def UNIQUAC(x,Rk,Qk,v,a,T):
    ri=np.sum(np.multiply(v,Rk),axis=1)
    qi=np.sum(np.multiply(v,Qk),axis=1)
    Ji=ri/(np.sum(np.multiply(ri,x)))
    Li=qi/(np.sum(np.multiply(qi,x)))
    gamc=1-Ji+np.log(Ji)-5*np.multiply(qi,(1-Ji/Li+np.log(Ji/Li)))
    eki=np.multiply(v,Qk)/qi
    tau=np.exp(-a/T)
    beta=eki*tau
    theta=np.sum(np.multiply(x,np.multiply(qi,eki)),axis=0)/np.sum(np.multiply(x,qi))
    s=theta*tau
    gamr=np.multiply(qi,1-np.sum(np.multiply(theta,beta)/s-np.multiply(eki,np.log(beta/s)),axis=1))
    gam=np.exp(gamc+gamr)
    return gam
def phisrk(T,P,Tc,Pc,w,y):
    Tr=T/Tc
    Pr=P/Pc
    m=0.48+1.547*w+0.176*np.multiply(w,w)
    a=np.multiply(1+np.multiply(m,1-np.sqrt(Tr)),1+np.multiply(m,1-np.sqrt(Tr)))
    Ai=0.42747*np.multiply(a,Pr/np.multiply(Tr,Tr))
    Bi=0.08664*Pr/Tr
    B=np.sum(np.multiply(y,Bi))
    ym=y*np.matrix.transpose(y)
    Aij=Ai*np.matrix.transpose(Ai)
    sAij=np.sqrt(Aij)
    A=np.sum(np.multiply(ym,sAij))
    def funz(xo):
        x1=xo-(xo**3-xo**2+(A-B-B**2)*xo-A*B)/(3*xo**2-2*xo+A-B-B**2)
        ercal=np.abs(x1-xo)
        while ercal>0.0001:
            xo=x1
            x1=xo-(xo**3-xo**2+(A-B-B**2)*xo-A*B)/(3*xo**2-2*xo+A-B-B**2)
            ercal=np.abs(x1-xo)
        return x1
    Z=funz(1)
    phi=np.exp((Z-1)*Bi/B-np.log(Z-B)-A/B*(2*np.sqrt(Ai)/np.sqrt(A)-Bi/B)*np.log((Z+B)/Z))
    return phi
def phil(T,P,Tc,Pc,w):
    Tr=T/Tc
    Pr=P/Pc
    m=0.48+1.547*w-0.176*np.multiply(w,w)
    a=np.multiply(1+np.multiply(m,1-np.sqrt(Tr)),1+np.multiply(m,1-np.sqrt(Tr)))
    om=0.08664
    psi=0.42748
    Ai=0.42747*np.multiply(a,Pr/np.multiply(Tr,Tr))
    Bi=om*Pr/Tr
    q=psi/om*a/Tr
    def funz(xo):
        retor=Bi+np.multiply(np.multiply(xo,xo+Bi),(1+Bi-xo)/(np.multiply(q,Bi)))-xo
        return retor
    def dfunz(xo):
        retor=2*xo/(np.multiply(q,Bi))-3*np.multiply(xo,xo)/np.multiply(q,Bi)+(Bi+np.multiply(Bi,Bi))/np.multiply(q,Bi)-1
        return retor
    def zetinha(xo):
        x1=xo-funz(xo)/dfunz(xo)
        ercal=np.sum(np.abs(x1-xo))
        while ercal>0.0001:
            xo=x1
            x1=xo-funz(xo)/dfunz(xo)
            ercal=np.sum(np.abs(x1-xo))
        return x1
    Z=zetinha(Bi)
    print(Z)
    phi=np.exp(Z-1-np.log(Z-Bi)-np.multiply(Ai/Bi,np.log((Z+Bi)/Z)))
    return phi
def Kf(x,y,T,P,Tc,Pc,w,Rk,Qk,v,a):
    phiv=phisrk(T,P,Tc,Pc,w,y)
    gamma=UNIQUAC(x,Rk,Qk,v,a,T)
    philo=phil(T,P,Tc,Pc,w)
    Ke=np.multiply(philo,gamma)/phiv
    return Ke
def philpr(T,P,Tc,Pc,w):
    Tr=T/Tc
    Pr=P/Pc
    m=0.37464+1.54226*w-0.176*np.multiply(w,w)
    a=np.multiply(1+np.multiply(m,1-np.sqrt(Tr)),1+np.multiply(m,1-np.sqrt(Tr)))
    om=0.0778
    psi=0.45724
    epsi=1-np.sqrt(2)
    sig=1+np.sqrt(2)
    Ai=psi*np.multiply(a,Pr/np.multiply(Tr,Tr))
    Bi=om*Pr/Tr
    q=psi/om*a/Tr
    def funz(xo):
        retor=Bi+np.multiply(np.multiply(xo+sig*Bi,xo+epsi*Bi),1+Bi-xo)/np.multiply(q,Bi)-xo
        return retor
    def dfunz(xo):
        Pro1=1+Bi-sig*Bi-epsi*Bi
        Pro2=sig*Bi+sig*np.multiply(Bi,Bi)+epsi*Bi+epsi*np.multiply(Bi,Bi)-sig*epsi*np.multiply(Bi,Bi)
        retor=(2*np.multiply(Pro1,xo)-3*np.multiply(xo,xo)+np.multiply(Pro2,xo))/np.multiply(q,Bi)-1
        return retor
    def zetinha(xo):
        x1=xo-funz(xo)/dfunz(xo)
        ercal=np.sum(np.abs(x1-xo))
        while ercal>0.000001:
            xo=x1
            x1=xo-funz(xo)/dfunz(xo)
            ercal=np.sum(np.abs(x1-xo))
        return x1
    Z=zetinha(Bi-0.001)
    print(Z)
    phi=np.exp(Z-1-np.log(Z-Bi)+np.multiply(Ai/(2*np.sqrt(2)*Bi),np.log((Z+epsi*Bi)/(Z+sig*Bi))))
    return phi
def gammacs(x,T,di,vil):
    vL=np.sum(np.multiply(x,vil))
    phi=np.multiply(x,vil)/vL
    R=1.987
    gamma=np.exp(np.multiply(vil,np.multiply(di-np.sum(np.multiply(phi,di)),di-np.sum(np.multiply(phi,di))))/(R*T)+np.log(vil/vL)+1-vil/vL)
    return gamma
def phigs(T,P,Tc,Pc,w):
    A0=2.05135;A1=-2.10899;A2=0;A3=-0.19396;A4=0.02282;A5=0.08852;A6=0;
    A7=-0.00872;A8=-0.00353;A9=0.00703;A10=-4.23893;A11=8.65808;
    A12=-1.2206;A13=-3.15224;A14=-0.025;
    Tr=T/Tc
    Pr=P/Pc
    logphio=A0+A1/Tr+A2*Tr+A3*np.multiply(Tr,Tr)+A4*np.multiply(Tr,np.multiply(Tr,Tr))+np.multiply(A5+A6*Tr+A7*np.multiply(Tr,Tr),Pr)+np.multiply(A8+A9*Tr,np.multiply(Pr,Pr))-np.log10(Pr)
    logphi1=A10+A11*Tr+A12/Tr+A13*np.multiply(Tr,np.multiply(Tr,Tr))+A14*(Pr-0.6)
    phi=np.float_power(10,logphio+np.multiply(w,logphi1))
    return phi