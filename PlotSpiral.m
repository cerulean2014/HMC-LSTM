close all; clear; clc;

%% Ground truth data construction
DataNum = 200;
a = 0;
b = 0.3;
z_cw = linspace(6*pi+1,1,DataNum);
r_cw = a + 50*b./z_cw;

x_cw = r_cw .* cos(z_cw) - 5;
y_cw = r_cw .* sin(z_cw);

z_cc = linspace(0,6*pi,DataNum);
r_cc = a + b * z_cc;
x_cc = r_cc .* cos(z_cc) + 5;
y_cc = r_cc .* sin(z_cc);

figure;
hold on;
plot(x_cw,y_cw);
plot(x_cc,y_cc);
hold off;
legend("clockwise","counter clockwise");
title("Ground truth plot");

%% Section 2: add Gausian noise
GN_std = 0.5;

%{
figure;
plot(x_cw,y_cw);
title("clockwise plot");

figure;
plot(x_cc,y_cc);
title("counter clockwise plot");
%}
cw = [x_cw;y_cw]';
cc = [x_cc;y_cc]';

noisy_cw = [x_cw + randn(size(x_cw)) * GN_std; y_cw + randn(size(y_cw)) * GN_std]';
noisy_cc = [x_cc + randn(size(x_cc)) * GN_std; y_cc + randn(size(y_cc)) * GN_std]';

cw = Normalise(cw);
cc = Normalise(cc);
noisy_cw = Normalise(noisy_cw);
noisy_cc = Normalise(noisy_cc);

separator = 10;
shorter_cw = cw(1:separator:end,:);
shorter_cc = cc(1:separator:end,:);


figure;
hold on;
scatter(noisy_cw(:,1),noisy_cw(:,2),'.');
scatter(noisy_cc(:,1),noisy_cc(:,2),'.');
hold off;
legend("clockwise","counter clockwise");
title("Noisy plot");

figure;
hold on;
scatter(noisy_cw(1:separator:end,1),noisy_cw(1:separator:end,2),'.');
scatter(noisy_cc(1:separator:end,1),noisy_cc(1:separator:end,2),'.');
hold off;
legend("clockwise","counter clockwise");
title("Noisy plot");



%% Section 3: save files

writematrix(cc,"counter_clockwise.csv");
%{
writematrix(cw,"clockwise.csv");
writematrix(noisy_cw,"noisy_clockwise.csv");
writematrix(noisy_cc,"noisy_counter_clockwise.csv");
writematrix(shorter_cw,"shorter_clockwise.csv");
writematrix(shorter_cc,"shorter_counter_clockwise.csv");
%}

function out = Normalise(data)
    out = data;
    temp = data(:,1);
    temp = temp - mean(temp);
    out(:,1) = temp / max(abs(temp));
    temp = data(:,2);
    temp = temp - mean(temp);
    out(:,2) = temp / max(abs(temp));
    out = out/2;
end
