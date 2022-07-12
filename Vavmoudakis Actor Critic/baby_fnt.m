function [dotx] = baby_fnt(t,x)

global percent amplitude Tf Pt UkUfinal Pfinal xfstate ufinal Wcfinal
global uDelayInterpFcn x1DelayInterpFcn x2DelayInterpFcn
global T uvec
global M R
global alphaa alphac
global A B Qxx Quu Qxu sigma UkU UkUdelay

Wc = [x(3);x(4);x(5);x(6);x(7);x(8)]; %critic
Wa = [x(9);x(10)]; %actor
p = x(11); %integral 

UkUfinal = [xfstate(1)^2 ; xfstate(1)*xfstate(2); xfstate(1)*ufinal; xfstate(2)^2 ; xfstate(2)*ufinal; ufinal^2];
Pfinal = 0.5*(xfstate)'*Pt*(xfstate); % equation 19

% Update control
ud = Wa'*(x(1:2)-xfstate); % equation 17

% State and control delays
uddelay = ppval(uDelayInterpFcn,t-T);
xdelay(1) = ppval(x1DelayInterpFcn,t-T);
xdelay(2) = ppval(x2DelayInterpFcn,t-T);

%Augmented state and Kronecker products
U = [x(1:2)-xfstate;ud-ufinal]; % Augmented state
Utone=U';
UkU = [U(1)^2 ; U(1)*U(2); U(1)*ud; U(2)^2 ; U(2)*ud; ud^2]; % U kron U
% This is not the exact UkronU (9x1 matrix), but we do it to lower the
% dimensionality. This way, Wc' would be 1x9, so the matrix multiplication
% can actually happen in ec.
UkUdelay = [xdelay(1)^2 ; xdelay(1)*xdelay(2); xdelay(1)*uddelay; xdelay(2)^2 ;...
    xdelay(2)*uddelay; uddelay^2];

Qxx = x(3:5); % equations 14-16
Quu = x(8);
Qxu = [x(6) x(7)]';
Qux = Qxu';

sigma = UkU - UkUdelay;

% integral reinforcement
dp =  0.5*((x(1:2)-xfstate)'*M*(x(1:2)-xfstate) -xdelay(1:2)*M*xdelay(1:2)' ...
    +ud'*R*ud - uddelay'*R*uddelay); 

% mine
QStar = Wc'*UkU;
uStar = -inv(Quu)*Qux*U(1:2);

% Approximation Errors 
ec = p + Wc'*UkU - Wc'*UkUdelay; % Critic approximator error
ecfinal = Pfinal - Wcfinal'*UkUfinal; % Critic approximator error final state
ea = Wa'*U(1:2)+.5*inv(Quu)*Qux*U(1:2); % Actor approximator error 
% or x(1:2) - xfstate instead of U(1:2) % or ud instead of Wa'*U(1:2)


% critic update
dWc = -alphac*((sigma./(sigma'*sigma+1).^2)*ec'+(UkUfinal./((UkUfinal'*UkUfinal+1).^2)*ecfinal')); % eq 22

% actor update
dWa = -alphaa*U(1:2)*ea'; % equation 23
   
% Persistent Excitation
if t<=(percent/100)*Tf
    unew=(ud+amplitude*exp(-0.009*t)*2*(sin(t)^2*cos(t)+sin(2*t)^2*cos(0.1*t)+...
        sin(-1.2*t)^2*cos(0.5*t)+sin(t)^5+sin(1.12*t)^2+cos(2.4*t)*sin(2.4*t)^3));
else
    unew=ud;
end

dx=A*[U(1);U(2)]+B*unew;
uvec = [uvec;unew];
dotx = [dx;dWc;dWa;dp;QStar;uStar]; % augmented state