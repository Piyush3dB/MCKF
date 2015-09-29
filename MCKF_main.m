clc;
clear;
format compact;

n  = 2;        %number of states
m  = 1;        %number of measurements
N  = 1000;     %total number of time steps
F  = [cos(pi/18),-sin(pi/18);sin(pi/18),cos(pi/18)];  %state transition matrix
H  = [1,1];    %observation matrix
s  = [0;0];    %initial true state
x  = [1;1];    %initial estimated state
P  = eye(2);   %initial state covariance

q1 = sqrt(0.01)*randn(N,1); %process noise
q2 = sqrt(0.01)*randn(N,1); %process noise
r  = sqrt(0.01)*randn(N,1); %measurement noise
Q  = diag([0.01,0.01]);     %covariance matrix of process noise
R  = 0.01;         %covariance matrix of measurement noise
sV = zeros(n,N);  %true state value
xV = zeros(n,N);  %estimated state value
zV = zeros(m,N);  %true measurement value
bV = zeros(1,N);  %iteration number

% simulate Kalman FIlter
for k=1:N

    s = F*s+[q1(k,1);q2(k,1)]; %true state update process
    
    z = H*s+r(k,1);            %true measurement update process

    [x,P,b] = MCKF(F,x,P,H,z,Q,R); % KF


    sV(:,k) = s;               %save true state value
    zV(:,k) = z;               %save true measurement value
    xV(:,k) = x;               %save estimated state value
    bV(:,k) = b;               %save iteration number
end


%plot results
figure;
plot(1:N,sV(1,:),'b-',1:N,xV(1,:),'r--');
L1=legend('true','MCKF',0);
xlabel('{\ittime step k}','Fontsize',15,'Fontname','Times new roman');
title('state estimate x(1)','Fontsize',15,'Fontname','Times new roman');

figure;
plot(1:N,sV(2,:),'b-',1:N,xV(2,:),'r--')
L1=legend('true','MCKF',0);
xlabel('{\ittime step k}','Fontsize',15,'Fontname','Times new roman');
title('state estimate x(2)','Fontsize',15,'Fontname','Times new roman');
