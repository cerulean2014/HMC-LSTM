close all; clear; clc;

NoNoiseX = readmatrix('PredictionOutputX0.csv');
NoNoiseY = readmatrix('PredictionOutputY0.csv');
SmallNoiseX = readmatrix('PredictionOutputX0.1.csv');
SmallNoiseY = readmatrix('PredictionOutputY0.1.csv');
MediumNoiseX = readmatrix('PredictionOutputX0.2.csv');
MediumNoiseY = readmatrix('PredictionOutputY0.2.csv');
LargeNoiseX = readmatrix('PredictionOutputX0.5.csv');
LargeNoiseY = readmatrix('PredictionOutputY0.5.csv');

%% Calculate all prediction entropies
Xedge = [-1:0.02:1];
Yedge = [-1:0.02:1];
Entropies = zeros(4,size(NoNoiseX,2));
for c = 1:size(NoNoiseX,2)
    Entropies(1,c) = Entropy(histogram2(NoNoiseX(2:end,c),NoNoiseY(2:end,c),'XBinEdges',Xedge,'YBinEdges',Yedge,'Normalization','probability'));
    Entropies(2,c) = Entropy(histogram2(SmallNoiseX(2:end,c),SmallNoiseY(2:end,c),'XBinEdges',Xedge,'YBinEdges',Yedge,'Normalization','probability'));
    Entropies(3,c) = Entropy(histogram2(MediumNoiseX(2:end,c),MediumNoiseY(2:end,c),'XBinEdges',Xedge,'YBinEdges',Yedge,'Normalization','probability'));
    Entropies(4,c) = Entropy(histogram2(LargeNoiseX(2:end,c),LargeNoiseY(2:end,c),'XBinEdges',Xedge,'YBinEdges',Yedge,'Normalization','probability'));
end
figure;
plot(Entropies');
legend('No noise','[-10%,10%] noise','[-20%,20%] noise','[-50%,50%] noise');
title("Prediction entropy measurements for different noise conditions");
xlabel('Samples');

%% Calculate all mean squared prediction errors
PredictionError = zeros(4,size(NoNoiseX,2));
GroundTruth = readmatrix('counter_clockwise.csv');
GroundTruth = GroundTruth(6:end,:)';

for c = 1:size(NoNoiseX,2)
    PredictionError(1,c) = (sum((NoNoiseX(2:end,c) - GroundTruth(1,c)).^2) + sum((NoNoiseY(2:end,c) - GroundTruth(2,c)).^2));
    PredictionError(2,c) = (sum((SmallNoiseX(2:end,c) - GroundTruth(1,c)).^2) + sum((SmallNoiseY(2:end,c) - GroundTruth(2,c)).^2));
    PredictionError(3,c) = (sum((MediumNoiseX(2:end,c) - GroundTruth(1,c)).^2) + sum((MediumNoiseY(2:end,c) - GroundTruth(2,c)).^2));
    PredictionError(4,c) = (sum((LargeNoiseX(2:end,c) - GroundTruth(1,c)).^2) + sum((LargeNoiseY(2:end,c) - GroundTruth(2,c)).^2));
end

%% Plot prediction error vs entropy
close all;
figure;
subplot 131;
En = Entropies(1,:);
Error = PredictionError(1,:);
scatter(En,Error,'.');
xlim([4 6.5]);
ylim([0 3000]);
xlabel('Prediction entropy');
ylabel('Mean squared error');
lsline;
title('SNR = +Inf');
subplot 132;
En = Entropies(2,:);
Error = PredictionError(2,:);
scatter(En,Error,'.');
xlim([4 6.5]);
ylim([0 3000]);
xlabel('Prediction entropy');
ylabel('Mean squared error');
lsline;
title('SNR = 10');
%{
subplot 143;
En = Entropies(3,:);
Error = PredictionError(3,:);
scatter(En,Error,'.');
xlim([4 6.5]);
xlabel('Prediction entropy');
ylabel('Mean squared error');
lsline;
title('SNR = 5');
%}
subplot 133;
En = Entropies(4,:);
Error = PredictionError(4,:);
scatter(En,Error,'.');
xlim([4 6.5]);
ylim([0 3000]);
xlabel('Prediction entropy');
ylabel('Mean squared error');
lsline;
title('SNR = 2');

%% Plot mean of predictions
figure;
subplot 131;
hold on;
plot(GroundTruth(1,:),GroundTruth(2,:));
plot(mean(NoNoiseX(2:end,:)),mean(NoNoiseY(2:end,:)),'.');
hold off;
xlim([-0.6 0.6]);
ylim([-0.6 0.6]);
legend('Ground Truth','Predictions');
xlabel('x');
ylabel('y');
title('SNR = +Inf');
subplot 132;
hold on;
plot(GroundTruth(1,:),GroundTruth(2,:));
plot(mean(SmallNoiseX(2:end,:)),mean(SmallNoiseY(2:end,:)),'.');
hold off;
xlim([-0.6 0.6]);
ylim([-0.6 0.6]);
legend('Ground Truth','Predictions');
xlabel('x');
ylabel('y');
title('SNR = 10');
%{
subplot 143;
hold on;
plot(GroundTruth(1,:),GroundTruth(2,:));
plot(mean(MediumNoiseX(2:end,:)),mean(MediumNoiseY(2:end,:)),'.');
hold off;
xlim([-0.6 0.6]);
ylim([-0.6 0.6]);
legend('Ground Truth','Predictions');
title('SNR = 5');
%}
subplot 133;
hold on;
plot(GroundTruth(1,:),GroundTruth(2,:));
plot(mean(LargeNoiseX(2:end,:)),mean(LargeNoiseY(2:end,:)),'.');
hold off;
xlim([-0.6 0.6]);
ylim([-0.6 0.6]);
legend('Ground Truth','Predictions');
xlabel('x');
ylabel('y');
title('SNR = 2');