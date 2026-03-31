function T0 = bvp_ic_GSEBM(x,T0_in,T_sp,Tx_sp)

% Guess for the solution of the BVP: 'T_sp'. Generally it is a spline
% passed from 'main_GSEBM.m'; and 'Tx_sp' is a derivative of that spline.
% As a simple option, the guess is a constant.

if T0_in % with a dummy 'Tx_sp' supplied
    T0 = [T0_in; 0];
else % with 'T0_in=0' supplied (not a dummy!)
    T0 = [ppval(T_sp,x); ppval(Tx_sp,x)];
end