%import static training data (700 patients) - this is the basis of the simple model
load('static_data_training.mat');
%The header variable contains the meaning of each column of static_train
%generate simple glm
%define Y = observations which should be loaded from clinical table
Y = static_train(:,2);

%define X = covariate matrix by taking features from table. 
%This currently only uses Gender as a covariate.
X = static_train(:,3:7);

%display distributions of each covariate    
%matlab starts with one not zero
disp('Gender distribution:');
disp(countcats(categorical(X(:,1))));


disp('Age distribution:');
disp(X(:,2));

disp('Respiratory comorbidities distribution:');
disp(X(:,3));

disp('Cardiovascular Comorbidities')
disp(X(:,4));

disp('Infection distribution:');
disp(X(:,5));


%should be mapped


%compute glm
%static model

[B,dev,stats] = glmfit(X,Y,'binomial', 'Link', 'logit');
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
disp(p_values);

%This code didn't work
%B is the coeffiencients 
%[beta, betaci] = coefCI(stats);
%df = stats.dfe;
%p_values = stats.p;

%repistory functiion and infection is the best value


%obtain the degrees of freedom and p-values for the estimated coefficients, you can access the stats output as follows:
%GLM output from the estimated coeffiecients 700 observations 694 error
%degrees of freedom p values 

%confidence intervals 


%plot phat versus patient along with its confidence bounds (1.96*stats.se)
Phat_CI = 1.96*stats.se;
figure(2)
errorbar(1:length(Phat), Phat, Phat_CI, 'b')
hold on
plot(Y(1:30),'r*')
title('Models for Each Patient')
Phat_LB = Phat - Phat_CI;






% plot first 30 patients prediction, uncertainty and labels.
figure(1)
plot(Phat(1:30))
hold on
% plot(Phat_LB,'b-')
% hold on
% plot(Phat_UB,'b-')
% hold on
plot(Y(1:30),'r*')
title('Models for Each Patient')



%%%%% Dynamic Model %%%%%
%choose less features and changing with time 
%label this the dynamic model save it as something else
[B_dynamic,dev_dynamic,stats_dynamic] = glmfit(X(:,[1,3]),Y,'binomial', 'Link', 'logit');
%construct phat from parameters and X 
Phat_dynamic = 1./(1+exp(-[ones(size(X(:,[1,3]),1),1) X(:,[1,3])]*B_dynamic)); %equivalent way to compute Phat
%Phat is the estimated probability of sepsis occurence for patients
dev_dynamic = sum(stats_dynamic.resid.^2)/stats_dynamic.dfe;
disp(['Deviance: ', num2str(dev_dynamic)]);





%static model
%test performance of models
 [threshold] = test_performance(Phat, Y);
 l=Y;
 [X,Y,T,AUC] = perfcurve(l,Phat, 1);
 figure;
 plot(X,Y)

%dynamic model
%test performance of models
 [threshold] = test_performance(Phat, Y);
% l=Y;
% [X,Y,T,AUC] = perfcurve(l,Phat, 1);
% figure;
% plot(X,Y)


%glm validation 
%%%%% Static Model Validation %%%%%
load('static_data_validation.mat');
Y_val = static_val(:,2);
X_val = static_val(:,3:7);
Phat_val = 1./(1+exp(-[ones(size(X_val,1),1) X_val]*B));
[threshold] = test_performance(Phat_val, Y_val);
l=Y_val;
[X,Y,T,AUC] = perfcurve(l,Phat_val, 1);
figure;
plot(X,Y)

%%%%% Dynamic Model Validation %%%%%
load('dynamic_data_validation.mat');
Y_val = dynamic_val(:,2);
X_val = dynamic_val(:,3:7);
Phat_val = 1./(1+exp(-[ones(size(X_val,1),1) X_val]*B_dynamic));
[threshold] = test_performance(Phat_val, Y_val);
l=Y_val;
[X,Y,T,AUC] = perfcurve(l,Phat_val, 1);
figure;
plot(X,Y)



