                        
%% Implementation of: Q-learning for continuous time linear systems:       
%% A model-free finite horizon optimal control approach (free final state)  
%% #### -------------------------------------------------------------- ####
clear all;
close all;
clc;

%% Dynamics and System Parameters
global A B C M R F n m Pt xfstate xfstateDist ufinal wfinal Wcfinal% Dynamics, initial and final conditions
global T Tf % ODE paramters
global alphaa alphaa2 alphac % GD parameters
global amplitude percent % Noise parameters
global uvec wvec uDelayInterpFcn wDelayInterpFcn x1DelayInterpFcn x2DelayInterpFcn

A  = [-1 0;0 -2];
B  = [0 1]';
C = [0 1]';
[n,m]=size(B); % States and controls
M  = diag([1,.1]);
R  = .1;
F = .1;
Pt= diag([.5,.5]); % Ricatti at final time

% ODEs parameters
T  = 0.05; % Delay
Tf = 6; % Finite horizon in seconds
N  = Tf/T; % Number of simulation rounds
 
% Gradient descent parameters
alphac=90;
alphaa=1.5;
alphaa2=1.5;

% Noise paramaters
amplitude = .1; % changes the amplitude of the PE (kyriakos pe)
percent = 50;

Wc0 = [rand(8,1);10*rand(2,1)]; % critic weights
Wa0 = .5*ones(4,1); % actor weights
u0= rand(m);
ufinal = 0.001;
wfinal = 0.001;
Wcfinal = 12*ones(10,1);

% Initial condition of the augmented system 
x0state = [32 -15]';  % initial state
xfstate = [-11.5 7.5]';  % final state
xfstateDist = [40 -25]';
% P0 = .1*ones(4,1); % TODO -> DELETE THIS

p0 = x0state'*M*x0state;  % integral RL, initil control 0 % from equation 8

% mine
QStar0 = rand(m);%648;%1000;
uStar0 = rand(m);%-1.68;%-26;
wStar0 = rand(m);

t_save_fnt = [];        % time vec
x_save_fnt = [x0state;Wc0;Wa0;p0;QStar0;uStar0;wStar0]';%;xfstate]'; % build the initial cond state vec. This is x initially

uDelayInterpFcn = interp2PWC(0,0,1);
wDelayInterpFcn = interp2PWC(0,0,1);
x1DelayInterpFcn = interp2PWC(x_save_fnt(:,1),0,0);
x2DelayInterpFcn = interp2PWC(x_save_fnt(:,2),0,0);

%% Solve ODEs

for i=1:N
    options = odeset('RelTol',1e-5,'AbsTol',1e-5,'MaxStep',.01); % 'OutputFcn',@odeplot,
    tic;
    sol_fnt = ode45(@distubBabyFnt,[(i-1)*T,i*T],x_save_fnt(end,:),options);
    toc;
    
    t_save_fnt = [t_save_fnt;sol_fnt.x'];    % save time
    x_save_fnt = [x_save_fnt;sol_fnt.y'];    % save state
   
    uDelayInterpFcn = interp2PWC(uvec,0,i*T);    % interpolate control input
    wDelayInterpFcn = interp2PWC(wvec,0,i*T);
    x1DelayInterpFcn = interp2PWC(x_save_fnt(:,1),0,i*T);
    x2DelayInterpFcn = interp2PWC(x_save_fnt(:,2),0,i*T);             
end

% Q=[Qxx; Qxu; Quu];
% 
% Qxx = [x(3:5)];
% Quu = x(8);
% Qxu = [x(6) x(7)]';

%% Plots
% x1, x2 to show at the reachability DID happen; it reached xfstate
figure 
set(gca,'FontSize',26); hold on;
plot(t_save_fnt,x_save_fnt(1:end-1,1),'LineWidth',2,'Color', 'b'); hold on;
plot(t_save_fnt,x_save_fnt(1:end-1,2),'LineWidth',2,'Color', 'm'); hold on;
xlabel('Time [s]'); ylabel('States'); legend('x_1','x_2');
grid on; hold off;

figure 
set(gca,'FontSize',26)
hold on;
for i=1:10
    plot(t_save_fnt,x_save_fnt(1:end-1,i+2),'-','LineWidth',2)
    hold on;
end
xlabel('Time [s]');ylabel('Critic Weights W_c');
legend('Wc_1','Wc_2','Wc_3','Wc_4','Wc_5','Wc_6, Wc_7, Wc_8, Wc_9, Wc_10');
grid on;
hold off;

% mine
figure 
set(gca,'FontSize',26)
hold on;
plot(t_save_fnt,x_save_fnt(1:end-1,13),'-','LineWidth',2)
plot(t_save_fnt,x_save_fnt(1:end-1,14),'-','LineWidth',2)
xlabel('Time [s]');ylabel('Actor Weights W_a');
legend('Wa_1','Wa_2');
grid on;
hold off;

% For the Q = Value function, either plot equation 19 or Q* before eq 16
% Equation 19 is a number
valueFunctionLastExpectedValue = xfstate'*Pt*xfstate;  % integral RL, initil control 0 % from equation 8
disp(valueFunctionLastExpectedValue)

valueFunctionLastActualValue = [x_save_fnt(end,1); x_save_fnt(end,2)]'*Pt*[x_save_fnt(end,1); x_save_fnt(end,2)];  % integral RL, initil control 0 % from equation 8
disp(valueFunctionLastActualValue)

% Q* before eq 16
figure 
set(gca,'FontSize',26)
hold on;
plot(t_save_fnt,x_save_fnt(1:end-1,16),'-','LineWidth',2)
xlabel('Time [s]');ylabel('Q*');
grid on;
hold off;
% FUCK YALL, IT ASYMPTOTICALLY CONVERGES, BADABEEM BADABOOM BADABAM

% u* from eq 15
figure 
set(gca,'FontSize',26)
hold on;
plot(t_save_fnt,x_save_fnt(1:end-1, 17),'-','LineWidth',2)
xlabel('Time [s]');ylabel('u*');
grid on;
hold off;

% w*
figure 
set(gca,'FontSize',26)
hold on;
plot(t_save_fnt,x_save_fnt(1:end-1, 18),'-','LineWidth',2)
xlabel('Time [s]');ylabel('w*');
grid on;
hold off;
