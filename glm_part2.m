%This code creates and tests a model of septic risk
clear;clc;
%import both clinical data and waveform data
load('static_data_training.mat');
load('dynamic_data_training.mat');

%%%%
%generate covariates for complex model

%preparing a septic/non-septic vector for glmfit
%note all septic data will be first, followed by all non-septic data
Y = nan(length(dynamic_train(:,1)),1);
X = nan(length(dynamic_train(:,1)),11);
X(:,6:11) = dynamic_train(:,3:8);
IDs = dynamic_train(:,1);%septic patient ID for each waveform time point
ID_uni = static_train(:,1);%patient's ID numbers

%create covariate matrix including both demographic info and waveform data
for i = 1:length(ID_uni)%for septic data
    ind = find(IDs==ID_uni(i));
    X(ind,1:5) = repmat(static_train(i,3:7),length(ind),1);
    display(num2str(i));
    Y(ind) = repmat(static_train(i,2),length(ind),1);
end

%%%%
%[B,dev,stats] = glmfit(X,Y,'normal');%find model parameters 
%Phat = 1./(1+exp(-[ones(size(X,1),1) X]*B)); %equivalent way to compute Phat
%[thresh] = test_performance(Phat, Y, "");

% Added code to identify parameters to use in model

figure_counter = 1;
distributions = ["binomial", "normal", "poisson"];
performances = [];

counter = 1;
max_display = 100;

for distribution = distributions
    disp("-------- NEW MODEL for...--------")
    disp(distribution)
    disp("---------------------------------")
    [B,dev,stats] = glmfit(X,Y, distribution);
    %construct phat from parameters and X 
    Phat = 1./(1+exp(-[ones(size(X,1),1) X]*B)); %equivalent way to compute Phat
    %Phat is the estimated probability of sepsis occurence for patients
    dev = sum(stats.resid.^2)/stats.dfe;
    disp(['Deviance: ', num2str(dev)]);
    
    %B is the coeffiencients 
    % Get standard errors of the coefficients from the stats structure
    SE = sqrt(diag(stats.covb)); % Standard errors of coefficients
    
    alpha = 0.05; % 95% confidence level
    t_critical = tinv(1 - alpha / 2, stats.dfe); % t-distribution critical value
    
    % Calculate confidence intervals for coefficients
    lower_bound = B - t_critical * SE;
    upper_bound = B + t_critical * SE;
    
    p_values = stats.p;
    disp("p-values...");
    disp(p_values);
    
    %Phat_UB = Phat + 1.96*stats.se;
    %Phat_LB = Phat - 1.96*stats.se;
    Phat_UB = 1./(1+exp(-[ones(size(X,1),1) X]*upper_bound));
    Phat_LB = 1./(1+exp(-[ones(size(X,1),1) X]*lower_bound));
    
    figure(figure_counter);
    plot(Phat(1:max_display));
    hold on;
    plot(Phat_LB(1:max_display));
    hold on;
    plot(Phat_UB(1:max_display));
    ylim([0.0, 1.0]);
    hold on;

    saveas(gcf, strcat(distribution, "_dynamic_finalized_preds.png"))
    
    figure_counter = figure_counter + 1;

    %figure(figure_counter);
    %plot(Phat(1:max_display));
    %hold on;
    %plot(Phat_LB(1:max_display));
    %hold on;
    %plot(Phat_UB(1:max_display));
    %hold on;
    %plot(Y(1:max_display))
    %ylim([0.0, 1.0]);
    %figure_counter = figure_counter + 1;

    [threshold] = test_performance(Phat, Y, strcat(distribution, "_dynamic"));
    disp("threshold...");
    disp(threshold);
    performances(counter) = threshold;
    counter = counter + 1;
end