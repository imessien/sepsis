%import static training data (700 patients) - this is the basis of the simple model
load('static_data_training.mat');
%The header variable contains the meaning of each column of static_train
%generate simple glm
%define Y = observations which should be loaded from clinical table
Y = static_train(:,2);

%define X = covariate matrix by taking features from table. 
%This currently only uses Gender as a covariate.
X = static_train(:,4:7); % UPDATES HERE

%display distributions of each covariate    
%matlab starts with one not zero
%disp('Gender distribution:');
%disp(countcats(categorical(X(:,1))));


%disp('Age distribution:');
%disp(X(:,2));

%disp('Respiratory comorbidities distribution:');
%disp(X(:,3));

%disp('Cardiovascular Comorbidities')
%disp(X(:,4));

%disp('Infection distribution:');
%disp(X(:,5));


% Make first glm fit using logistic distribution
figure_counter = 1;
distributions = ["binomial", "normal", "poisson"];
performances = [];

counter = 1;
max_display = 30;

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

    disp("coefficient estimates:");
    disp(B);
    disp("lower bounds:");
    disp(lower_bound);
    disp("upper bounds:");
    disp(upper_bound);
    
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

    saveas(gcf, strcat(distribution, "_finalized_preds.png")) % UPDATES HERE
    
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

    [threshold] = test_performance(Phat, Y, distribution);
    disp("threshold...");
    disp(threshold);
    performances(counter) = threshold;
    counter = counter + 1;
end


