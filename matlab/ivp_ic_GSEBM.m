function T0 = ivp_ic_GSEBM(x0_data,T0_data,x)

% Initial condition (ic) for the initial value problem (ivp) regarding the
% Ghil-Sellers (GS) energy balance climate model (EBM); see 'main_GSEBM.m'.

if length(T0_data) == 1 % uniform temperature (with dummy 'x0_data')
    T0 = T0_data; 
elseif  T0_data(1) == 0 % IC to satisfy boundary 
    T0 = spline(x0_data,T0_data,x);
else
    T0 = pchip(x0_data,T0_data,x);
end