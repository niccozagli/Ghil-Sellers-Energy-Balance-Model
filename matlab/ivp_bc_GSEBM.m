function [pl,ql,pr,qr] = ivp_bc_GSEBM(xl,Tl,xr,Tr,t,k1_sp,k2_sp,c4,c5)

% Boundary condition (bc) for the initial value problem (ivp) regarding the
% Ghil-Sellers (GS) energy balance climate model (EBM) (see
% 'main_GSEBM.m'): vanishing slopes of the temperature profile at the
% poles. A workaround to force these BCs is needed because of the
% singularity at the poles:
% http://www.mathworks.de/support/solutions/en/data/1-TFCHU/index.html?product=ML

% Vectorised code suitable for ensemble simulation.

n  = length(Tl);
pl = zeros(n,1);
ql = 1/cos(pi*xl/2)./(ppval(k1_sp,xl) + ...
    ppval(k2_sp,xl)*c4./Tl.^2.*exp(-c5./Tl));
pr = zeros(n,1);
qr = 1/cos(pi*xr/2)./(ppval(k1_sp,xr) + ...
    ppval(k2_sp,xr)*c4./Tr.^2.*exp(-c5./Tr));