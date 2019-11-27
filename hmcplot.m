% THIS IS THE ONE TO SHOW!

% Some reference from https://theclevermachine.wordpress.com/2012/11/18/mcmc-hamiltonian-monte-carlo-a-k-a-hybrid-monte-carlo/
% https://pythonhosted.org/pyhmc/ 
% EXAMPLE 2: HYBRID MONTE CARLO SAMPLING -- Bivarite Normal
close all; clear; clc;

rng default;

% STEP SIZE
epsilon = 0.3;
N = 10^4;
L = 10;
rho = [1 0.8;0.8,1]; % Variance-Covariance matrix
A = inv(rho);

[X,Y] = meshgrid(linspace(-5,5,1000));
GaussianP = mvnpdf([X(:),Y(:)],[0,0],rho);
GaussianP = reshape(GaussianP,size(X));


% Define potential energy function for this bivariate Gaussian
U = @(x) (x.'*A*x + log((2*pi)^2*det(rho)));

% Define the gradient of this potential energy
dU = @(x) transpose(x)*A;

% Define kinetic energy function
K = @(p) sum((transpose(p)*p))/2;

% Initialise first state
x = zeros(2,N);
Boundary = 5;
x(:,1) = [rand*Boundary;rand*Boundary];

t = 1;
while t < N
    t = t + 1;

    % Sample momentum
    p0 = randn(2,1);

    %% SIMULATE HAMILTONIAN DYNAMICS
    %{
    The Leap Frog method updates the momentum and position variables
    sequentially, starting by simulating the momentum dynamics over a small
    interval of time \delta /2, then simulating the position dynamics over
    a slightly longer interval in time \delta, then completing the momentum
    simulation over another small interval of time delta/2 so that x and p
    now exist at the same point in time;
    The Leap Fog method can be run for L steps to simulate dynamics over
    L * delta units of time. This particular discretization method has a
    number of properties that make it preferable to other approximation
    methods like Euler¡¯s method, particularly for use in MCMC or in dynamic systems.
    %}
    
    % Total L steps
    % First 1/2 step in p
    pStar = p0 - epsilon*dU(x(:,t-1))'/2;

    % Make a step in x
    xStar = x(:,t-1) + epsilon*pStar;

    % Running one steps in leap frog
    for j = 1:(L-1)
        pStar = pStar - epsilon*dU(xStar)';
        xStar = xStar + epsilon*pStar;
    end

    % Last half step for p
    pStar = pStar - epsilon/2*dU(xStar)';

    % COULD NEGATE MOMENTUM HERE TO LEAVE
    % THE PROPOSAL DISTRIBUTION SYMMETRIC.
    % HOWEVER WE THROW THIS AWAY FOR NEXT
    % SAMPLE, SO IT DOESN'T MATTER

    % EVALUATE ENERGIES AT
    % START AND END OF TRAJECTORY
    U0 = U(x(:,t-1));
    UStar = U(xStar);

    K0 = K(p0);
    KStar = K(pStar);

    % MH
    dH = exp((U0 + K0) - (UStar + KStar));
    alpha = min(1,dH);

    u = rand;
    if dH == 1 || u < alpha 
        x(:,t) = xStar;
    else
        x(:,t) = x(:,t-1);
    end
end

figure;
subplot 121
mesh(X,Y,GaussianP);
colorbar;
view([0 0 1]);
title('Target 2D Gaussian Distribution');
StateNum = 50;

subplot 122
mesh(X,Y,GaussianP);
colorbar;
view([0 0 1]);
title('HMC samples');
hold on;
scatter3(x(1,:),x(2,:),ones(1,length(x(1,:))),'g.');
hold off;
%plot(x(1,1:StateNum),x(2,1:StateNum),'ro-','Linewidth',2);
xlim([-Boundary Boundary]); ylim([-Boundary Boundary]);
%legend({'Samples',strcat('First ',num2str(StateNum),'States')},'Location','Best')

figure;
scatter(x(1,1001:end),x(2,1001:end),'.');
xlim([-Boundary Boundary]); ylim([-Boundary Boundary]);
title('HMC samples without 10% warm-up');
xlabel('x1');
ylabel('x2');
%{
figure;
histogram2(x(1,:),x(2,:),50,'Normalization','probability');
xlim([-5 5]);
ylim([-5 5]);
title('2D histogram of HMC samples');
xlabel('x1');ylabel('x2');zlabel('P');
%}