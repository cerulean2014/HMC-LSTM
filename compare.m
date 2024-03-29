close all; clear; clc;

D = readmatrix('counter_clockwise.csv');
D = D(6:end,:)';
NoNoiseX = readmatrix('PredictionOutputX0.csv');
NoNoiseY = readmatrix('PredictionOutputY0.csv');
SmallNoiseX = readmatrix('PredictionOutputX0.1.csv');
SmallNoiseY = readmatrix('PredictionOutputY0.1.csv');
MediumNoiseX = readmatrix('PredictionOutputX0.2.csv');
MediumNoiseY = readmatrix('PredictionOutputY0.2.csv');
LargeNoiseX = readmatrix('PredictionOutputX0.5.csv');
LargeNoiseY = readmatrix('PredictionOutputY0.5.csv');

N = [mean(NoNoiseX(2:end,:));mean(NoNoiseY(2:end,:))];
S = [mean(SmallNoiseX(2:end,:));mean(SmallNoiseY(2:end,:))];
M = [mean(MediumNoiseX(2:end,:));mean(MediumNoiseY(2:end,:))];
L = [mean(LargeNoiseX(2:end,:));mean(LargeNoiseY(2:end,:))];

R0 = readmatrix('PX0.csv');
R1 = readmatrix('PX0.1.csv');
R2 = readmatrix('PX0.25.csv');
R5 = readmatrix('PX0.5.csv');

errors = [0 0 0 0];
Gerrors = [0 0 0 0];

figure;
subplot 221;
hold on;
scatter(N(1,:),N(2,:),'.');
scatter(R0(1,:),R0(2,:),'.');
plot(D(1,:),D(2,:));
hold off;
legend('Expectation','GD','Ground Truth');
xlabel('x');
ylabel('y');
errors(1) = sum(sum((N-D).^2));
Gerrors(1) = sum(sum((R0-D).^2));
xlim([-0.6 0.6]);
ylim([-0.6 0.6]);
title('SNR = +Inf');
subplot 222;
hold on;
scatter(S(1,:),S(2,:),'.');
scatter(R1(1,:),R1(2,:),'.');
plot(D(1,:),D(2,:));
Gerrors(2) = sum(sum((R1-D).^2));
hold off;
legend('Expectation','GD','Ground Truth');
xlabel('x');
ylabel('y');
errors(2) = sum(sum((S-D).^2));
title('SNR = 10');
subplot 223;
hold on;
scatter(M(1,:),M(2,:),'.');
scatter(R2(1,:),R2(2,:),'.');
plot(D(1,:),D(2,:));
hold off;
legend('Expectation','GD','Ground Truth');
xlabel('x');
ylabel('y');
Gerrors(3) = sum(sum((R2-D).^2));
errors(3) = sum(sum((M-D).^2));
title('SNR = 5');
subplot 224;
hold on;
scatter(L(1,:),L(2,:),'.');
scatter(R5(1,:),R5(2,:),'.');
plot(D(1,:),D(2,:));
hold off;
legend('Expectation','GD','Ground Truth');
xlabel('x');
ylabel('y');
errors(4) = sum(sum((L-D).^2));
Gerrors(4) = sum(sum((R5-D).^2));
title('SNR = 2');