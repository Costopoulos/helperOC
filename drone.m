function [data, tau2] = drone()
%% Grid
grid_min = [-8; -8; -5; -3]; % Lower corner of computation domain
grid_max = [8; 8; 5; 3];    % Upper corner of computation domain
N = 21*ones(4,1);         % Number of grid points per dimension

g = createGrid(grid_min, grid_max, N);

%% target set
R = 3;
data0 = shapeCylinder(g, [2, 4], [0; 0; 0; 0], R); %x-z

%% time vector
t0 = 0;
tMax = 3;
dt = 0.1;
tau = t0:dt:tMax;

%% problem parameters
uMode = 'max';

%% Pack problem parameters
dCar = Quad6D([0,0,0,0]);

% Put grid and dynamic systems into schemeData
schemeData.grid = g;
schemeData.dynSys = dCar;
schemeData.accuracy = 'low'; %set accuracy
schemeData.uMode = uMode;

%% additive random noise
% HJIextraArgs.addGaussianNoiseStandardDeviation = [0; 0; 0.5; 0];
% Try other noise coefficients, like:
%    [0.2; 0; 0]; % Noise on X state
%    [0.2,0,0;0,0.2,0;0,0,0.5]; % Independent noise on all states
%    [0.2;0.2;0.5]; % Coupled noise on all states
%    {zeros(size(g.xs{1})); zeros(size(g.xs{1})); (g.xs{1}+g.xs{2})/20}; % State-dependent noise

%% Compute value function
HJIextraArgs.visualize = true;
HJIextraArgs.fig_num = 1;
HJIextraArgs.deleteLastPlot = true;
HJIextraArgs.stopConverge = true;
HJIextraArgs.makeVideo = true;

[data, tau2] = HJIPDE_solve(data0, tau, schemeData, 'zero', HJIextraArgs);

deriv = computeGradients(g, data);

%% Compute optimal trajectory from some initial state
compTraj = true;
if compTraj
  
  %set the initial state
  xinit = [-4.8; 0; -0.52; -0.6];
  
  %check if this initial state is in the BRS/BRT
  %value = eval_u(g, data, x)
  value = eval_u(g,data(:,:,:,:,end),xinit);
  if value <= 0 %if initial state is in BRS/BRT
    % find optimal trajectory
    
    dCar.x = xinit; %set initial state of the dubins car

    TrajextraArgs.uMode = uMode; %set if control wants to min or max
    TrajextraArgs.dMode = 'max';
    TrajextraArgs.visualize = true; %show plot
    TrajextraArgs.fig_num = 2; %figure number
    
    %we want to see the dimensions (x and z)
    TrajextraArgs.projDim = [1 0 1 0]; % computes optimal traj and visualizes
    
    %flip data time points so we start from the beginning of time
    dataTraj = flip(data,4);
    
    % [traj, traj_tau] = ...
    % computeOptTraj(g, data, tau, dynSys, extraArgs)
    [traj, traj_tau] = ...
      computeOptTraj(g, dataTraj, tau2, dCar, TrajextraArgs);
  
    figure(6)
    clf
    h = visSetIm(g, data(:,:,:,:,end));
    hold on
    s = scatter3(xinit(1), xinit(2), xinit(3));
    s.SizeData = 70;
    title('The reachable set at the end and xinit')
    hold off
  
    %plot traj
    figure(4)
    plot(traj(1,:), traj(2,:))
    hold on
    xlim([-8 8])
    ylim([-8 8])
    % add the target set to that
    [g2D, data2D] = proj(g, data0, [0 1 0 1]);
    visSetIm(g2D, data2D, 'green');
    title('2D projection of the trajectory & target set')
    hold off
  else
    error(['Initial state is not in the BRS/BRT! It has a value of ' num2str(value,2)])
  end
end

end
