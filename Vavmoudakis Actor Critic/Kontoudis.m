function [currentState, time] = Kontoudis(initializations, currentState, xfstate)%, uvec)
global uvec
global wvec

%% Decrypt Init Vars
T = initializations{8};
N = initializations{10};

time = [];

%% Solve ODE
options = odeset('RelTol',1e-5,'AbsTol',1e-5,'MaxStep',.01);

for i=1:N
    sol_fnt = ode45(@(t,x) actorCritic(t,x,xfstate,initializations),[(i-1)*T,i*T],currentState(end,:),options);
    
    time = [time;sol_fnt.x'];    % save time
    currentState = [currentState;sol_fnt.y'];    % save state
    
    uDelayInterpFcn = interp2PWC(uvec,0,i*T);    % interpolate control input
    wDelayInterpFcn = interp2PWC(wvec,0,i*T);
    x1DelayInterpFcn = interp2PWC(currentState(:,1),0,i*T);
    x2DelayInterpFcn = interp2PWC(currentState(:,2),0,i*T);
    delays = {uDelayInterpFcn; wDelayInterpFcn; x1DelayInterpFcn; x2DelayInterpFcn};
    initializations(end) = {delays};
end
end
