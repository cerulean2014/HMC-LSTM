close all; clear; clc;

NoNoiseX = readmatrix('PredictionOutputX0.csv');
NoNoiseY = readmatrix('PredictionOutputY0.csv');
SmallNoiseX = readmatrix('PredictionOutputX0.1.csv');
SmallNoiseY = readmatrix('PredictionOutputY0.1.csv');
MediumNoiseX = readmatrix('PredictionOutputX0.2.csv');
MediumNoiseY = readmatrix('PredictionOutputY0.2.csv');
LargeNoiseX = readmatrix('PredictionOutputX0.5.csv');
LargeNoiseY = readmatrix('PredictionOutputY0.5.csv');

%% Plot histogram for a certain prediction value
close all;
col = 1;
Xedge = [-0.4:0.02:0.4];
Yedge = [-0.4:0.02:0.4];

En = zeros(1,3);
TopMarker = 0.03;
figure;
subplot 141;
h = histogram2(NoNoiseX(2:end,col),NoNoiseY(2:end,col),'XBinEdges',Xedge,'YBinEdges',Yedge,'FaceColor','flat','Normalization','probability');
view([0 0 1]);
hold on;
plot3(NoNoiseX(1,col),NoNoiseY(1,col),TopMarker,'ro');
hold off;
colorbar;
caxis([0 TopMarker]);
title("No noise")
En(1) = Entropy(h);
subplot 142;
h = histogram2(SmallNoiseX(2:end,col),SmallNoiseY(2:end,col),'XBinEdges',Xedge,'YBinEdges',Yedge,'FaceColor','flat','Normalization','probability');
view([0 0 1]);
hold on;
plot3(SmallNoiseX(1,col),SmallNoiseY(1,col),TopMarker,'ro');
hold off;
colorbar;
caxis([0 TopMarker]);
title("Small noise");
En(2) = Entropy(h);
subplot 143;
h = histogram2(MediumNoiseX(2:end,col),MediumNoiseY(2:end,col),'XBinEdges',Xedge,'YBinEdges',Yedge,'FaceColor','flat','Normalization','probability');
view([0 0 1]);
hold on;
plot3(MediumNoiseX(1,col),MediumNoiseY(1,col),TopMarker,'ro');
hold off;
colorbar;
caxis([0 TopMarker]);
title("Medium noise");
En(3) = Entropy(h);
subplot 144;
h = histogram2(LargeNoiseX(2:end,col),LargeNoiseY(2:end,col),'XBinEdges',Xedge,'YBinEdges',Yedge,'FaceColor','flat','Normalization','probability');
view([0 0 1]);
hold on;
plot3(LargeNoiseX(1,col),LargeNoiseY(1,col),TopMarker,'ro');
hold off;
colorbar;
caxis([0 TopMarker]);
title("Large noise");
En(4) = Entropy(h);
%En

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

%% Compare mean prediction with ground truth
D = readmatrix('counter_clockwise.csv');
figure;
subplot 221
hold on;
scatter(mean(NoNoiseX(2:end,:)),mean(NoNoiseY(2:end,:)),'.');
plot(D(:,1),D(:,2));
hold off;
xlim([-0.6 0.6]);
ylim([-0.6 0.6]);
title("Predictions at SNR = +Inf");
legend("Expectation of prediction","Ground truth");
subplot 222
hold on
scatter(mean(SmallNoiseX(2:end,:)),mean(SmallNoiseY(2:end,:)),'.');
plot(D(:,1),D(:,2));
hold off;
xlim([-0.6 0.6]);
ylim([-0.6 0.6]);
title("Predictions at SNR = 10 (trained)");
legend("Expectation of prediction","Ground truth");
subplot 223
hold on
scatter(mean(MediumNoiseX(2:end,:)),mean(MediumNoiseY(2:end,:)),'.');
plot(D(:,1),D(:,2));
hold off;
xlim([-0.6 0.6]);
ylim([-0.6 0.6]);
title("Predictions at SNR = 5");
legend("Expectation of prediction","Ground truth");
subplot 224
hold on
scatter(mean(LargeNoiseX(2:end,:)),mean(LargeNoiseY(2:end,:)),'.');
plot(D(:,1),D(:,2));
hold off;
xlim([-0.6 0.6]);
ylim([-0.6 0.6]);
title("Predictions at SNR = 2");
legend("Expectation of prediction","Ground truth");