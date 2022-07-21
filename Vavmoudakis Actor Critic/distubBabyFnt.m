function [dotx] = distubBabyFnt(t,x)

global percent amplitude Tf Pt UkUfinal Pfinal xfstate xfstateDist ufinal wfinal Wcfinal
global uDelayInterpFcn wDelayInterpFcn x1DelayInterpFcn x2DelayInterpFcn
global T uvec wvec
global M R F
global alphaa alphac
global A B C Qxx Quu Qxu Qww Qxw sigma UkU UkUdelay

Wc = [x(3);x(4);x(5);x(6);x(7);x(8);x(9);x(10);x(11);x(12)]; %critic
Wa = [x(13);x(14);x(15);x(16)]; %actor
p = x(17); %integral 

UkUfinal = [xfstate(1)^2 ; xfstate(1)*xfstate(2); xfstate(1)*ufinal; xfstate(1)*wfinal; ...
            xfstate(2)^2 ; xfstate(2)*ufinal; xfstate(2)*wfinal; ...
            ufinal^2; ufinal*wfinal; ...
            wfinal^2];
Pfinal = 0.5*(xfstate)'*Pt*(xfstate); % equation 19

% Update control
ud = Wa(1:2)'*(x(1:2)-xfstate); % equation 17
wd = Wa(3:4)'*(x(1:2)-xfstate); % equation 17

% State and control delays
uddelay = ppval(uDelayInterpFcn,t-T);
wddelay = ppval(wDelayInterpFcn,t-T);
xdelay(1) = ppval(x1DelayInterpFcn,t-T);
xdelay(2) = ppval(x2DelayInterpFcn,t-T);

%Augmented state and Kronecker products
U = [x(1:2)-xfstate;ud-ufinal;wd-wfinal]; % Augmented state
Utone=U';
UkU = [U(1)^2 ; U(1)*U(2); U(1)*ud; U(1)*wd; ...
       U(2)^2 ; U(2)*ud; U(2)*wd; ...
       ud^2; ud*wd; ...
       wd^2]; % U kron U
% This is not the exact UkronU (9x1 matrix), but we do it to lower the
% dimensionality. This way, Wc' would be 1x9, so the matrix multiplication
% can actually happen in ec.
UkUdelay = [xdelay(1)^2 ; xdelay(1)*xdelay(2); xdelay(1)*uddelay; xdelay(1)*wddelay;...
            xdelay(2)^2 ; xdelay(2)*uddelay; xdelay(2)*wddelay; ...
            uddelay^2; uddelay*wddelay; ...
            wddelay^2];

Qxx = x(3:6); % equations 14-16
Quu = x(11);
Qww = x(12);
Qxu = [x(6) x(7)]';
Qux = Qxu';
Qxw = [x(8) x(9)]';
Qwx = Qxw';

sigma = UkU - UkUdelay;

% integral reinforcement
% dp =  0.5*((x(1:2)-xfstate)'*M*(x(1:2)-xfstate) -xdelay(1:2)*M*xdelay(1:2)' ...
%     +ud'*R*ud - uddelay'*R*uddelay);

dp = 0.5*((x(1:2)-xfstate)'*M*(x(1:2)-xfstate) -xdelay(1:2)*M*xdelay(1:2)' ...
    +ud'*R*ud - uddelay'*R*uddelay - wd'*F*wd + wddelay'*F*wddelay); 

% mine
QStar = Wc'*UkU;
uStar = -inv(Quu)*Qux*U(1:2);
wStar = inv(Qww)*Qwx*U(1:2);

% Approximation Errors 
ec = p + Wc'*UkU - Wc'*UkUdelay; % Critic approximator error
ecfinal = Pfinal - Wcfinal'*UkUfinal; % Critic approximator error final state
ea1 = Wa(1:2)'*U(1:2)+.5*inv(Quu)*Qux*U(1:2); % Actor approximator error 
ea2 = Wa(3:4)'*U(1:2)-.5*inv(Qww)*Qwx*U(1:2);
% ea = Wa'*U(1:2)+.5*inv(Quu)*Qux*U(1:2)-.5*inv(Qww)*Qwx*U(1:2);
% or x(1:2) - xfstate instead of U(1:2) % or ud instead of Wa'*U(1:2)
% BASICALLY, looking at eq30, ud = Wa'*U(1:2) = Wa'(x(1:2)-xfstate) wants
% to reach u* = inv(Quu)*Qux*U(1:2), that's why ea is defined like this.
% ud is slowly getting to u*, aka
% Wa'*U(1:2) -> inv(Quu)*Qux*U(1:2)

% critic update
dWc = -alphac*((sigma./(sigma'*sigma+1).^2)*ec'+(UkUfinal./((UkUfinal'*UkUfinal+1).^2)*ecfinal')); % eq 22

% actor update
dWa1 = -alphaa*U(1:2)*ea1'; % equation 23
dWa2 = -alphaa*U(1:2)*ea2'; % equation 23
dWa = [dWa1; dWa2];
% dWa = max([dWa1 dWa2]);
% dWa = -alphaa*U(1:2).*ea';

% Persistent Excitation
if t<=(percent/100)*Tf
    unew=(ud+amplitude*exp(-0.009*t)*2*(sin(t)^2*cos(t)+sin(2*t)^2*cos(0.1*t)+...
        sin(-1.2*t)^2*cos(0.5*t)+sin(t)^5+sin(1.12*t)^2+cos(2.4*t)*sin(2.4*t)^3));
    wnew=(wd+amplitude*exp(-0.009*t)*2*(sin(t)^2*cos(t)+sin(2*t)^2*cos(0.1*t)+...
        sin(-1.2*t)^2*cos(0.5*t)+sin(t)^5+sin(1.12*t)^2+cos(2.4*t)*sin(2.4*t)^3));
else
    unew=ud;
    wnew=wd;
end

dx=A*[U(1);U(2)]+B*unew+C*wnew;
uvec = [uvec;unew];
wvec = [wvec;wnew];
dotx = [dx;dWc;dWa;dp;QStar;uStar;wStar]; % augmented state