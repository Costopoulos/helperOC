function reachabilityMain()
global uvec
close all;
clc;

%% Dynamics and System Parameters
A  = [-1 0;0 -2];
B  = [0 1]';
[~,m] = size(B); % States and controls
M  = diag([1,.1]);
R  = .1;
Pt = diag([.5,.5]); % Ricatti at final time

% ODEs parameters
T  = 0.05; % Delay
Tf = 6; % Finite horizon in seconds
N  = Tf/T; % Number of simulation rounds
 
% Gradient descent parameters
alphac = 90;
alphaa = 1.5;

% Noise paramaters
amplitude = .1; % changes the amplitude of the PE (kyriakos pe)
percent = 50;

Wc0 = [rand(5,1);10*rand(1)]; % critic weights
Wa0 = .5*ones(2,1); % actor weights
ufinal = 0.001;
Wcfinal = 12*ones(6,1);

% Initial condition of the augmented system 
x0state = [32 -15]';  % initial state
xfstate = [-11.5 7.5]';  % final state

p0 = x0state'*M*x0state;  % integral RL, initil control 0 % from equation 8

QStar0 = rand(m);
uStar0 = rand(m);

x_save_fnt = [x0state;Wc0;Wa0;p0;QStar0;uStar0]'; % build the initial cond state vec. This is x initially

uDelayInterpFcn = interp2PWC(0,0,1);
x1DelayInterpFcn = interp2PWC(x_save_fnt(:,1),0,0);
x2DelayInterpFcn = interp2PWC(x_save_fnt(:,2),0,0);

delays = {uDelayInterpFcn; x1DelayInterpFcn; x2DelayInterpFcn};
initializations = {A; B; M; R; Pt; T; Tf; N; ...
                   alphaa; alphac; amplitude; ...
                   percent; ufinal; Wcfinal; delays};

%% Set target
[xVal, yVal] = circle(-11.5,7.5,4);%(0,0,8);
figure 
hold on;
plot(xVal,yVal, 'r');
plot(x0state(1),x0state(2),'.b')
xlim([-20 40])
ylim([-20 40])
title('Initial Point(Blue) and Target(Red)')
pause(0.1);
hold off;

% Retrieve actual number of points of the target
[~, xPoints] = size(xVal); % no need for yPoints cause they are the same

allTrajectoriesVector = [];
batchEndSignifier = zeros(1,2); 
% Append zeros to the end of each batch in order to know when a trajectory 
% for a specific xfstate ended. We only care for the states of each 
% stateVector

%% Call Kontoudis' algorithm
for i=1:xPoints
    xfstate = [xVal(i) yVal(i)]';
    tic
    uvec = [];
    [stateVector, time] = Kontoudis(initializations, x_save_fnt, xfstate);
    allTrajectoriesVector = [allTrajectoriesVector; stateVector(:,1:2); batchEndSignifier];
    toc
end

%% Pre-Plot
% create the vector with all the final points in order to have the
% reachable set
allFinalStatesVector = [];
for i=1:length(allTrajectoriesVector)
    if (allTrajectoriesVector(i,:) == 0) % if batchEndSignifier is found
        allFinalStatesVector = [allFinalStatesVector; allTrajectoriesVector(i-1,:)];
    end
end


%% Plots
figure
hold on;
plot(xVal,yVal, 'r'); hold on; % target
plot(x0state(1),x0state(2),'.b'); hold on; % initial State
plot(allFinalStatesVector(1:end,1),allFinalStatesVector(1:end,2),'-b'); hold on;
xlim([-20 40])
ylim([-20 40])
title('Reachable Set')
grid on; hold off;

figure
hold on;
plot(xVal,yVal, 'r'); hold on;
plot(allFinalStatesVector(1:end,1),allFinalStatesVector(1:end,2),'-b'); hold on;
plot(x0state(1),x0state(2),'.b'); hold on;
% Find Optimal Trajectory
[closestPoint, trajectoryPlot] = findOptimalTrajectory(x0state,...
                                 allTrajectoriesVector,allFinalStatesVector);
% actually plot
plot(closestPoint, '.y'); hold on;
plot(trajectoryPlot(1:end,1), trajectoryPlot(1:end,2), 'Color', 'g'); hold on;
xlim([-20 40])
ylim([-20 40])
title('Optimal Trajectory to Reachable Set')
grid on; hold off;

% x1, x2 to show at the reachability DID happen; it reached xfstate
figure 
set(gca,'FontSize',26); hold on;
plot(time,stateVector(1:end-1,1),'LineWidth',2,'Color', 'b'); hold on;
plot(time,stateVector(1:end-1,2),'LineWidth',2,'Color', 'm'); hold on;
xlabel('Time [s]'); ylabel('States'); legend('x_1','x_2');
grid on; hold off;

figure 
set(gca,'FontSize',26)
hold on;
for i=1:6
    plot(time,stateVector(1:end-1,i+2),'-','LineWidth',2)
    hold on;
end
xlabel('Time [s]');ylabel('Critic Weights W_c');
legend('Wc_1','Wc_2','Wc_3','Wc_4','Wc_5','Wc_6');
grid on;
hold off;

% mine
figure 
set(gca,'FontSize',26)
hold on;
plot(time,stateVector(1:end-1,9),'-','LineWidth',2)
plot(time,stateVector(1:end-1,10),'-','LineWidth',2)
xlabel('Time [s]');ylabel('Actor Weights W_a');
legend('Wa_1','Wa_2');
grid on;
hold off;

% For the Q = Value function, either plot equation 19 or Q* before eq 16
% Equation 19 is a number
valueFunctionLastExpectedValue = xfstate'*Pt*xfstate;  % integral RL, initil control 0 % from equation 8
disp(valueFunctionLastExpectedValue)

valueFunctionLastActualValue = [stateVector(end,1); stateVector(end,2)]'*Pt*[stateVector(end,1); stateVector(end,2)];  % integral RL, initil control 0 % from equation 8
disp(valueFunctionLastActualValue)

% Q* before eq 16
figure 
set(gca,'FontSize',26)
hold on;
plot(time,stateVector(1:end-1,12),'-','LineWidth',2)
xlabel('Time [s]');ylabel('Q*');
grid on;
hold off;

% u* from eq 15
figure 
set(gca,'FontSize',26)
hold on;
plot(time,stateVector(1:end-1, 13),'-','LineWidth',2)
xlabel('Time [s]');ylabel('u*');
grid on;
hold off;
end

function [xVal, yVal] = circle(x,y,r)
% Creates a target circle and returns its points
% x and y are the coordinates of the center of the circle
% r is the radius of the circle
% 0.01 is the angle step, bigger values will draw the circle faster but
% you might notice imperfections (not very smooth)
ang = linspace(0,2*pi,100);
xp = r*cos(ang);
yp = r*sin(ang);
xVal = x+xp;
yVal = y+yp;
end

function [closestPoint, trajectoryToPlot] = findOptimalTrajectory(initState,...
                                            trajectoriesVector, ...
                                            finalStateVector)
% Function to return the optimal trajectory to the reachable set
% param{initState} is the initial state from which we are starting
% param{trajectoriesVector} is the vector that contains all trajectories from the
% initState to each xfstate
% param{finalStateVector} is the vector that contains all the final states of the
% trajectories in trajectoriesVector

% find closest point of the target to the initial point
dist = sqrt((initState(1) - finalStateVector(:,1)).^2 + (initState(2) - finalStateVector(:,2)).^2);
[~, indexOfMin] = min(dist);
% indexOfMin contains the information of which batch went closest to the
% reachable set
closestX = finalStateVector(indexOfMin, 1);
closestY = finalStateVector(indexOfMin, 2);
closestPoint = [closestX closestY];

trajectoryPlot = [];
% find which batch to plot
start = 1;
batchCount = 0;
for i=1:length(trajectoriesVector)
    if (trajectoriesVector(i,:) == 0)
        batchCount = batchCount + 1;
        if batchCount == indexOfMin
            trajectoryToPlot = [trajectoryPlot; trajectoriesVector(start:i-1, :)];
            break;
        end
        start = i+1;
    end
end
end
