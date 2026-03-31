function dydx = bvp_eq_GSEBM(x,y,Q_sp,b_sp,z_sp,...
    k1_sp,k2_sp,k1x_sp,k2x_sp,c1,c2,c3,c4,c5,sig,m1,um,mu,alpha_max)

% Definition of the ODE deriving from the PDE of the Ghil-Sellers (GS)
% energy balance climate model (EBM) with vanishing tendency. We are
% looking for stationary states by solving a boundary value problem (BVP);
% see 'main_GSEBM.m'.

% Albedo
alpha = ppval(b_sp,x) - c1*(um + min(y(1,:) - c2*ppval(z_sp,x) - um,0)); 
% Introduce upper and lower cutoffs, in a vectorized manner.
alpha(alpha > alpha_max) = alpha_max;
alpha(alpha < 0.25) = 0.25;

g = c4./y(1,:).^2.*exp(-c5./y(1,:));
k = ppval(k1_sp,x) + ppval(k2_sp,x).*g;
% Net radiative energy transport 
F = mu*ppval(Q_sp,x).*(1 - alpha) - ...
    sig*y(1,:).^4.*(1 - m1*tanh(c3*y(1,:).^6)); 

dydx = [y(2,:);
    -(pi/2)^2*F./k - ...
    (-pi/2*min([tan(pi/2*x); ones(size(x))*1e10]) + ...
    (ppval(k1x_sp,x) + ppval(k2x_sp,x).*g)./k).*y(2,:) - ...
    (ppval(k2_sp,x).*g./y(1,:).^2.*(c5 - y(1,:)/2)./k).*y(2,:).^2];
        % cot function is singular at 0 (meantioned by Ghil (1976) on the
        % right of p.8) and it is not cancelled by multiplying by a zero
        % y(2) due to the boundary, so we needed to use a cheap trick.