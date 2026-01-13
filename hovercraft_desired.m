function ydot = hovercraft_desired(y,k)
global kn;
% y=[u,v,p,r,x,y,phi,psi]';
%%%%%%%%%%%%计算出y的一阶导数，然后通过龙哥库塔法求解出y的值%%%%%%%%%%%%%
phi=y(3);    psi=y(4);  
u=30*0.514;    v=0;  p=0; r=0/57.3; %直航1
if k>kn
r=0.5*tanh(0.006*(k-kn))/57.3;%回转1
end
% disp('rd')
% disp(r*57.3)
      ydot=[   u*cos(psi)-v*sin(psi)*cos(phi);
               u*sin(psi)+v*cos(psi)*cos(phi);
               p;
               r*cos(phi) ];        
end
