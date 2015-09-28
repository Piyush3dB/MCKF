function [x,P,b]=MCKF(F,x,P,H,z,Q,R)
% mckf maximum correntropy Kalman filter
% [x,P,b]=mckf(F,x,P,H,z,Q,R) 
% Inputs:  F: state transition matrix
%          x: optimal estimated state of last time step
%          P: estimated state covariance of last time step          
%          H: observation matrix
%          z: measurement of current time step
%          Q: covariance matrix of process noise
%          R: covariance matrix of measurement noise
%          b: iteration number
% Outputs: x: optimal estimated state of current time step
%          P: estimated state covariance of current time step
%

n  = numel(x);   %number of states
m  = numel(z);   %number of measurements
x1 = F*x;        %priori estimated state 
P1 = F*P*F'+Q;   %priori estimated state covariance
B_all = [P1,zeros(n,m);zeros(m,n),R];
B  = (chol(B_all))';
D  = inv(B)*[x1;z];
W  = inv(B)*[eye(n);H];
x3 = x1;         
x2 = x1+ones(n,1);
b=0;             %initial iteration number

delta = 2;       %kernel bandwidth

% Now iterate
while (norm((x3-x2)/norm(x2))>=1e-6)&&(b<=10000)
    x2 = x3;  
    e  = D - W*x2;      %error
    
    for a = 1:n+m
        G(a) = exp(-(e(a,:)^2)/(2*delta^2));
    end

    C   = diag(G);
    P11 = B(1:n,1:n)*inv(C(1:n,1:n))*B(1:n,1:n)';
    R1  = B((n+1):(n+m),(n+1):(n+m))*inv(C((n+1):(n+m),(n+1):(n+m)))*B((n+1):(n+m),(n+1):(n+m))';
    K1  = P11*H'*inv(H*P11*H'+R1);
    x3  = x1+K1*(z-H*x1);
    b   = b+1;
end

x = x3;      %optimal estimated state of current time step
P = (eye(n)-K1*H)*P1*(eye(n)-K1*H)'+K1*R*K1'; %estimated state covariance of current time step
