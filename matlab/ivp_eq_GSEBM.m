function [c,f,s] = ivp_eq_GSEBM(x,t,T,Tx,C_sp,Q_sp,b_sp,z_sp,...
    k1_sp,k2_sp,c1,c2,c3,c4,c5,sig,m1,um,mu,alpha_max)

% Definition of the PDE of the Ghil-Sellers (GS) energy balance climate
% model (EBM) for the purpose of solving an initial value problem (ivp) by
% executing 'main_GSEBM.m'.

% Vectorised code suitable for ensemble simulation.

% Albedo
alpha = ppval(b_sp,x) - c1*(um + min(T - c2*ppval(z_sp,x) - um,0)); 
% Introduce upper and lower cutoffs, in a vectorized manner.
alpha(alpha > alpha_max) = alpha_max;
alpha(alpha < 0.25) = 0.25;

trig = cos(pi*x/2);
c = ppval(C_sp,x).*trig.*ones(size(T));
    % In case of a system of eqs. as for an ensemble simulation, we have to
    % replicate values by multiplying by 'ones' because in the simulation
    % process only a single 'x' value is passed.
f = cos(pi*x/2).*(ppval(k1_sp,x) + ...
    ppval(k2_sp,x)*c4./T.^2.*exp(-c5./T)).*Tx*(2/pi)^2;
    % The sign of divergence term does not change (-1*-1=+1)
s = (mu*ppval(Q_sp,x).*(1 - alpha) - ...
    sig*T.^4.*(1 - m1*tanh(c3*T.^6))).*trig;