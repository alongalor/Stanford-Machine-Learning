%% Part I: Initialization

clear ; close all; clc

%% Part II: Loading Data

%  The following code will load the dataset into MATLAB.
load ('X.mat');
load ('Y.mat');

%% Part III: Data Cleaning

% Deleting features with zero stdv
apples = [];
for i=1:size(X,2)
    if std(X(:,i)) == 0
        apples(end+1) = i;
    end
end
X(:,apples) = [];

% Data Cleaning
clear apples i

%% Part IV: Split Data into Training and Cross Validation Sets

% Cross Validation
cv = cvpartition(length(X),'holdout',0.40);

% Training Set
Xtrain = X(training(cv),:);
Ytrain = Y(training(cv),:);

% Training Set
Xval = X(test(cv),:);
Yval = Y(test(cv),:);

% Data Cleaning
clear X Y cv

%% Part V: Feature Normalization and Transformations

% Map X onto Polynomial Features and Normalize
% Xtrain = x2fx(Xtrain,'quadratic');               % Apply Transformations
[Xtrain_norm, mu, sigma] = featureNormalize(Xtrain);           % Normalize
Xtrain_norm = [ones(size(Xtrain_norm, 1), 1), Xtrain_norm];    % Add Ones

% Map X_poly_val and normalize (using mu and sigma)
% Xval = x2fx(Xval,'quadratic');                   % Apply Transformations
Xval_norm = bsxfun(@minus, Xval, mu);
Xval_norm = bsxfun(@rdivide, Xval_norm, sigma);                % Normalize
Xval_norm = [ones(size(Xval_norm, 1), 1), Xval_norm];          % Add Ones

%% Part VI: Cross Validation: Selecting Lambda

[lambda_vec, error_train, error_val] = ...
    validationCurve(Xtrain_norm, Ytrain, Xval_norm, Yval);

close all;
plot(lambda_vec, error_train, lambda_vec, error_val)
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');

fprintf('lambda\t\tTrain Error\tValidation Error\n');
for i = 1:length(lambda_vec)
	fprintf(' %f\t%f\t%f\n', ...
            lambda_vec(i), error_train(i), error_val(i));
end

[minerror i] = min(error_val);
minerror
minlambda = lambda_vec(i)