% Main simulation file for the Ghil-Sellers (GS) energy balance climate
% model (EBM) described by Michael Ghil (1976). Out of the several versions
% of the model described in the cited paper, 'Climate Stability for a
% Sellers-Type Model', the one specified by Eq. (4) is implemented here.
% Successful completion of the simulation results in the solution to the
% initial value problem (IVP): the zonal-average 'air' TEMPERATURE
% (extrapolated to sea level) as a function of latitude and time T(x,t).

% http://dx.doi.org/10.1175/1520-0469(1976)033<0003:CSFAST>2.0.CO;2

% The partial differential equation (PDE) of the EBM reads as follows:
%
% c(x)*T't = (2/pi)^2/sin(pi*x/2)*[sin(pi*x/2)*[k1(x)+g(T)*k2(x)]*T'x]'x -
% sigma*T^4*[1 - m*tanh(c3*T^6)] + mu*Q(x)*[1-albedo(x,T)]
%
% where ()'t and ()'x stand for differentiation wrt. time and latitude,
% respectively. Boundary conditions (BC) of vanishing slopes at the poles
% are imposed. The T-dependence of the albedo reflects the ICE-ALBEDO
% POSITIVE FEEDBACK, which can realize bistability for some parameter
% settings.

% The boundary value problem (BVP) for the asymptotic stationary state,
% defined by an ordinary differential equation (ODE), is solved to verify
% the result. Note that with this method the unstable stationary state,
% when it exists due to the arising bistability of the model, can also be
% found. This code is provided as a supplementary material to the paper
% 'Global instability in the Ghil-Sellers model' by Bodai et al. (2014), in
% which another method for finding the unstable solution, the edge-tracking
% technique, is described and applied to the G-S EBM.

% The code is ready to be executed -- with a default setup -- by hitting
% the 'run' button. Model and simulation parameters can be changed below.
% -- Read the descriptions of the parameters, and the comments on possible
% issues with the robustness of the numerics for certain settings.

% Tamas Bodai, 28.04.2014. 

clear 
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Model/simulation setup. Set desired values to the following variables --
% within the delimiter bars (lines filled with '%' characters).
% Manipulation of e.g. the coarsegraining that can be done further below
% requires a deeper understanding of the model and numerical procedures.

% Relative solar strength; mu=1 represents present-day conditions
mu = 1.;  
% Maximal value for the albedo
albedo_max = 0.6; 

% Final time of simulation. Run an appropriately long simulation, depending
% on the initial conditions, in order to approximate well the steady state!
t_f = 1e9;

% Initial condition (IC), i.e., temperature profile, for the IVP. As a
% simple option a scalar value can be provided representing uniform
% temperature. Set 'T0_ivp=0' to use a present-day observational
% temperature profile; or, if something else is desired, then assign new
% values to 'T0_data' and 'x0_data' further below.
T0_ivp = 280;

% Initial/guess temperature profile for the BVP. As a simple option a
% scalar value can be provided representing uniform temperature. Set
% 'T0_bvp = 0' to use a perturbed version of the solution to the IVP; or,
% if something else is desired, then assign new values to 'T_sp' further
% below.
T0_bvp = 0; 

% Choose one of three options for the differentiation of the guess
% temperature profile for the BVP: 1 for 'fnder', 2 for 'pdeval', 3 for
% 'gradient'. 
mdgtp = 1; 

% Case switch for removing negative values of 'k22' (1) or not (0). Read
% this for details:
% http://www.mombu.com/science/physics/t-negative-diffusion-coefficient-numerically-2962366.html
% Note: e.g. with mu=1; albedo_max=0.85; t_f=1e9; T0_ivp=280; T0_bvp=0;
% mdgtp=1; cefvs=1; rmnvk2=0 (in an attempt to reproduce Ghil's results)
% 'bvp4c' is not able to yield a solution (reason unknown, although
% changing to cefvs=0 or mdgtp=2 or mdgtp=3 might give a clue). With
% rmnvk2=1 (ceteris paribus) we can have a solution also to the BVP.
rmnvk2 = 1;

% Case switch for constraining empirical functions of latitude to have
% vanishing slopes at the boundaries (1) or not (0).
cefvs  = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Empirical functions of latitude. -- Since only 'sample values' are
% available, inter- and extrapolation will be necessary for producing 
% values at any other latitude.

% Sample values of the latitude at which data points are available.
% Although the model is symmetric to the equator (as a result of averaging
% the observational data for the northern and southern hemispheres at
% corresponding +ve and -ve latitudes), the range of latitude considered
% extends from pole-to-pole, because with the simplified setup for a single
% hemisphere only -- by turning symmetry conditions into boundary
% conditions -- the BCs cannot be satisfied (reason unknown). The latitude
% coordinate is normalized/nondimensionalized so that the full range is
% [-1,1].
x1 = [-90:10:-10 0:10:90]/90; 
x2 = [-85:10:-5  5:10:85]/90; % no data point on equator

% Available values of empirical functions at latitudes (on just one
% hemisphere to start with) as given by 'x1' or 'x2' above.
% Values at 'x1':
% Combined air-land-sea effective heat capacity [cal/cm^2/K]
C = [500 1000 1500 4725 5625 5812 5813 5625 6000 5625];
% High-frequency solar irradiance [cal/cm^2/sec]
Q = [0.426 0.440 0.484 0.579 0.696 0.804 0.894 0.961 1.003 1.017]*1e-2;
% Values at 'x2':
% 
b2 = [2.912 2.96 2.934 2.914 2.915 2.868 2.821 2.804 2.805]; % [-]
    % The value 2.192 given by Ghil (1976) is suspected to be a typo, and
    % so changed here to the plausible value 2.912. With this the
    % temperature at x=-1,1 corresponds very well with Fig 1.a of Ghil
    % (1976), but not so much with 2.192.
% c2*z: difference between 'sea level' and ground air temperatures 
z2 = [1204.5 820 295 150.5 193.5 301 261 133.5 156]; % [m]
% Eddy diffusivity coefficients; k1*T'x: sensible heat flux,
% k2*g(T)*T'x: latent heat flux
k12 = [0.47113 0.61988 1.19933 1.50214 1.51063 1.69562 2.02342 3.20611 ...
    4.80401]*1e-5; % [cal/K/cm^2/sec]
if rmnvk2
    % Remove -ve values of k22 so that k=k1+g*k2 would not be -ve near the
    % equator for mu >= 1.024 (with albedo_max=0.85, rmnvk2=0, cefvs=0);
    % and also choose the first entry suitably so as to prevent -ve values
    % at the pole after extrapolation.
    k22 = [0.3 0.9314 1.9772 3.4348 4.8316 3.7359 0.6903 0.2 0.1]*1e-2; 
else
    k22 = [0 0.9314 1.9772 3.4348 4.8316 3.7359 0.6903 -2.5401 ...
        -10.5975]*1e-2; % [cal/dyn/sec]
end

% 'Mirror' data vectors
C   = [C   fliplr(C(1:end-1))];
Q   = [Q   fliplr(Q(1:end-1))];
b2  = [b2  fliplr(b2)];
z2  = [z2  fliplr(z2)];
k12 = [k12 fliplr(k12)];
k22 = [k22 fliplr(k22)];

if cefvs
    % Interpolate between grids given by 'x1' and 'x2', so that we can use
    % 'x1' only. This preliminary interpolation (in fact: also
    % extrapolation) is needed to define values on the boundaries, where we
    % want to introduce constrains. We use an interpolation algorithm
    % (pchip) for this, different from the one (spline) used below for data
    % fitting. This is to respect monotonicity in the data (see help file
    % for pchip).
    b1  = pchip(x2,b2, x1); 
    z1  = pchip(x2,z2, x1);
    k11 = pchip(x2,k12,x1);
    k21 = pchip(x2,k22,x1);
    % When fitting by a cubic spline, we constrain the splines to have zero
    % slopes at the boundaries, just like the BC for the temperature. 
    C   = [0 C   0];
    Q   = [0 Q   0];
    b   = [0 b1  0];
    z   = [0 z1  0];
    k1  = [0 k11 0];
    k2  = [0 k21 0];
    % Pass the splines to the pde specification functions (e.g.
    % ivp_eq_GSEBM), i.e., don't use the function 'spline' every time of
    % evaluating the rhs. of the eqn.
     C_sp = spline(x1,C  );
     Q_sp = spline(x1,Q  );
     b_sp = spline(x1,b  );
     z_sp = spline(x1,z  );
    k1_sp = spline(x1,k1 );
    k2_sp = spline(x1,k2 );
else
     C_sp = spline(x1,C  );
     Q_sp = spline(x1,Q  );
     b_sp = spline(x2,b2 );
     z_sp = spline(x2,z2 );
    k1_sp = spline(x2,k12);
    k2_sp = spline(x2,k22);
end

% Constants
c1 = 0.009; % [1/K]
c2 = 0.0065; % [K/m]
c3 = 1.9e-15; % [K^-6]
c4 = 6.105*0.75*exp(19.6)*1e0; % [dyn*K/cm^2] 
    % redundant 1e3 factor given by Ghil (1976) removed
c5 = 5350; % [K]
sig = 1.356e-12; % [cal/cm^2/sec/K^4]
m1 = 0.5; % [-]
um = 283.16; % A certain yearly average temperature [K]

% Definition (function handle) of the PDE of the EBM
ivp_eq = @(x,t,T,Tx)ivp_eq_GSEBM(x,t,T,Tx,C_sp,Q_sp,b_sp,z_sp,...
    k1_sp,k2_sp,c1,c2,c3,c4,c5,sig,m1,um,mu,albedo_max);
m  = 0;
    % Note: m=0 specifies no symmetry in the problem. Taking m=1 specifies
    % cylindrical symmetry, while m=2 specifies spherical symmetry.
    
% Definition of the BCs
ivp_bc = @(xl,ul,xr,ur,t)ivp_bc_GSEBM(xl,ul,xr,ur,t,k1_sp,k2_sp,c4,c5);

% Definition of the ICs by sample values assigned by the user to 'T_data'
% with corresponding latitudes 'x_data'.
% Ghil (1976) provides the following observational data for present-day
% conditions, which can actually be used as an IC:
T_data = [247.3625 252.0740 262.5715 271.2980 278.9325 ...
    285.7530 291.4090 296.0815 298.7815 299.3510]; % [K]
% 'Mirror' data vector
T_data = [T_data fliplr(T_data(1:end-1))];
% Optionally: enforce the BCs already at the start
%T_data = [0 T_data fliplr(T_data(1:end-1)) 0];
if ~T0_ivp 
    T0_ivp = T_data;
end
x0_data = x1;

ivp_ic = @(x)ivp_ic_GSEBM(x0_data,T0_ivp,x);

% Definition of the grid for the coarsegrained numerical solution. Because
% of the singularity at the pole, a workaround for imposing the BCs is
% needed, as described here:
% http://www.mathworks.de/support/solutions/en/data/1-TFCHU/index.html?product=ML
delta = 1e-3;
    % 'delta' has to be tuned to the 'length' of 'xgrid'
xgrid = [-1+delta*[1 1.01], ... for the workaround
    linspace(-1+2*delta,1-2*delta,201), ... % main part of the grid
          1-delta*[1.01 1]];  % for the workaround

% Time span of simulation
tspan = linspace(0,t_f,101); % [sec]
    % The function 'pdepe' performs the time integration with an ODE solver
    % that selects both the time step and integrator scheme
    % dynamically/adaptively. The elements of 'tspan' merely specify the
    % time instants at which we want to have solution data, and the
    % computational effort depends weakly on the length of 'tspan'.
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
% Simulation: solution to the IVP.
T_ivp = pdepe(m,ivp_eq,ivp_ic,ivp_bc,xgrid,tspan); % Temperature [K]

% Plot the solution
figure;
surf(xgrid,tspan,T_ivp);
title('Solution of the Ghil-Sellers EBM');
xlabel('latitude x [-]');
ylabel('time t [sec]');
zlabel('Temperature T [K]');
set(findall(gcf,'type','text'),'fontSize',16,'fontWeight','bold')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Solve the BVP in order to verify the numerical solution to the IVP
% obtained by 'pdepe'. Mind that with this method we can find also the
% unstable solution of this bistable model. Another technique, the
% edge-tracking technique, is described by Bodai et al. (2014).

% Refined sample values of latitude for producing data defining the BVP
x_samp = linspace(x1(1),x1(end),1001);

% The IC can be a perturbed version of the numerical solution to the IVP.
% Use e.g. the cos function which has vanishing slopes at the boundaries.
Ap = 10; % perturbation 'amplitude'
T_sp = spline(xgrid,[0 T_ivp(end,:)+Ap*cos(pi*xgrid) 0]);

% We need to supply its derivative too:
switch mdgtp
    case 1
        Tx_sp = fnder(T_sp,1);
    case 2
        Tx_sp = spline(x_samp,gradient(ppval(T_sp,x_samp),...
            x_samp(2)-x_samp(1)));
    case 3
        [T_out,Tx_out] = pdeval(m,xgrid,T_ivp(end,:),xgrid);
        Tx_sp = spline(xgrid,Tx_out-Ap*sin(pi*xgrid));
            % Don't forget the perturbation!
end

if T0_bvp % then make dummy assignments
    T_sp = 0;
    Tx_sp = 0;
end

bvp_ic = @(x)bvp_ic_GSEBM(x,T0_bvp,T_sp,Tx_sp);
solinit = bvpinit(xgrid,bvp_ic);

% Definition (function handle) of the ODE as for the BVP
% First, differentiate splines fitted to empirical data. Let's use the same
% switch as for the temperature profile!
if mdgtp == 1
    k1x_sp = fnder(k1_sp,1);
    k2x_sp = fnder(k2_sp,1);
else
    k1x_sp = spline(x_samp,gradient(ppval(k1_sp,x_samp),...
        x_samp(2)-x_samp(1)));
    k2x_sp = spline(x_samp,gradient(ppval(k2_sp,x_samp),...
        x_samp(2)-x_samp(1)));
end

bvp_eq = @(x,y)bvp_eq_GSEBM(x,y,Q_sp,b_sp,z_sp,...
    k1_sp,k2_sp,k1x_sp,k2x_sp,c1,c2,c3,c4,c5,sig,m1,um,mu,albedo_max);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Solve the BVP
sol = bvp4c(bvp_eq,@bvp_bc_GSEBM,solinit);
T_bvp = deval(sol,xgrid);

% Plot the two numerical solutions to the IVP and BVP in one diagram
figure; plot(xgrid,T_ivp(end,:),xgrid,T_bvp(1,:))
xlabel('latitude x [-]');
ylabel('Temperature T [K]');
legend('IVP','BVP')
set(findall(gcf,'type','text'),'fontSize',16,'fontWeight','bold')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% A diagnostic plot: meridional heat transfer rate specified to half the
% global surface area [W/m^2]
[c,f,s] = ivp_eq(xgrid,0,T_bvp(1,:),T_bvp(2,:));
figure; plot(xgrid,-f*pi/2*1/0.239*1e4)
xlabel('latitude x [-]');
ylabel('j [W/m^2]');
set(findall(gcf,'type','text'),'fontSize',16,'fontWeight','bold')