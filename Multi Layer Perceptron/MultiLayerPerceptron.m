%% Multi-layer perceptron
classdef MultiLayerPerceptron < matlab.mixin.SetGet % ('hgsetget' for previous versions)
    
    % Properties (public access)
    properties (Access = 'public')
        
        % Multi-layer perceptron data
        
        LengthsOfLayers            = [];          % Lengths of layers (inputs, hidden, outputs)
        Weights                    = {};          % Weights, grouped by layers (cells of matrices)
        Biases                     = {};          % Biases, grouped by layers (cells of vectors)
        HiddenActFcn               = 'tanh';      % Activation function of hidden layers
        OutputActFcn               = 'softmax';   % Activation function of output layer   
        
        % Training general parameters
        
        TrainingAlgorithm          = 'GD';        % Training algorithm
        SplitRatio                 = 0.7;         % Ratio between training and validation sets      
        NumberOfEpochs             = 200;         % Maximum number of epochs
        MinimumMSE                 = 1e-3;        % Minimum mean square error
        UpdateFcn                  = []           % Update function
        
        % Gradient descent algorithm parameters
        
        LearningRate               = 0.15;        % Gradient descent learning rate
        SizeOfBatches              = 32;          % Size of batches
        Momentum                   = 0.9;         % Gradient descent momentum (relevant if SizeOfBatches equal to 1)
        
      
    end
    
    % Properties (private set access)
    properties(SetAccess = 'private')
    
        Outputs                    = [];          % Outputs
        CurrentEpoch               = 0;           % Current epoch
        MeanSquareErrors           = [];          % Mean square errors for training until current epoch
        ValidationMeanSquareErrors = [];          % Mean square errors for validation until current epoch
        TrainingStep               = '';          % Training step
    
    end
    
    % Properties (public access)
    properties (Access = 'private')
        
        % Multi-layer perceptron data
        NumberOfLayers            = 0;            % Number of layers (inputs, hidden, outputs)
        Values                    = {};           % Values of layers 
        Errors                    = {};           % Errors of layers
        HiddenActivationFunction  = [];           % Hidden activation function
        HiddenDerivative          = [];           % Hidden activation function derivative
        OutputActivationFunction  = [];           % Output activation function
        OutputDerivative          = [];           % Output activation function derivative
        
        % Gradient descent algorithm data
        PreviousWeights            = {};          % Previous weights for momentum
        PreviousBiases             = {};          % Previous biases, for momentum
        GradientsWeights           = {};          % Gradients for weights, for gradient descent by batch
        GradientsBiases            = {};          % Gradients for biases, for gradient descent by batch  
        
        % Levenberg-Marquardt algorithm data
        WeightsVector              = [];          % Weights/biases vector
        JacobianMatrix             = [];          % Jacobian matrix, applied to weights/biases vector
        
    end
    
    % Properties (constant)
    properties (Constant)
        
        dw = 1e-6;                                % Finite difference step for jacobian matrix calculation
        
    end
    
    % Methods (public)
    methods (Access = 'public')
        
        % Method 'MultiLayerPerceptron'
        function Object = MultiLayerPerceptron(varargin)
            
            for v = 1:2:length(varargin)
                
                Property = varargin{v};
                Value = varargin{v+1};
                set(Object,Property,Value);
                
            end
            
        end
        
        % Method 'set'
        function Object = set(Object,varargin)
            
            Properties = varargin(1:2:end);
            Values     = varargin(2:2:end); %#ok<PROPLC>
            
            for n = 1:length(Properties)
                
                [Indicator, Property] = Object.isprop(Properties{n});
                
                % Property control
                if Indicator              
                    Object.(Property) = Values{n}; %#ok<PROPLC>
                else
                    error('No appropriate property ''%s''for class ''MultiLayerPerceptron''.',Properties{n});
                end
                
                % Value control
                switch lower(Property)
                    
                    % Update function
                    case lower('UpdateFcn')
                    
                        switch class(Values{n}) %#ok<PROPLC>
                        
                            case 'char'
                            
                                switch lower(Values{n}) %#ok<PROPLC>
                                    case {'','none'}, Object.UpdateFcn = @(~)true;
                                    case 'default',   Object.UpdateFcn = @(MLP)updateFigure(MLP);
                                    otherwise, error('MyComponent:incorrectType',...
                                                    ['Property ''UpdateFcn'' shall be:\n',...
                                                     ' - String ''None''         (or '''' or []),\n',...
                                                     ' - String ''Default''      (internal GUI),\n',...
                                                     ' - User defined function (boolean = UpdateFcn(MLP))']);
                                end
                                
                            case 'function_handle', Object.UpdateFcn = Values{n}; %#ok<PROPLC>
                            
                            otherwise, error('MyComponent:incorrectType',...
                                             ['Property ''UpdateFcn'' shall be:\n',...
                                              ' - String ''None''         (or '''' or []),\n',...
                                              ' - String ''Default''      (internal GUI),\n',...
                                              ' - User defined function (boolean = UpdateFcn(MLP))']);
                        end
                        
                    % Hidden layers activation function
                    case lower('HiddenActFcn')
                        
                        switch lower(Values{n}) %#ok<PROPLC>
                            case {'linear','logistic','relu','softmax','tanh'}
                            otherwise, error('MyComponent:incorrectType',...
                                             ['Property ''HiddenActFcn'' shall be chosen among:\n',...
                                              ' - ''linear'',\n',...
                                              ' - ''logistic'',\n',...
                                              ' - ''ReLU'',\n',...
                                              ' - ''tanh''']);
                        end
                        
                    % Output layer activation function
                    case lower('OutputActFcn')
                     
                        switch lower(Values{n}) %#ok<PROPLC>
                            case {'linear','logistic','relu','tanh','softmax'}
                            otherwise, error('MyComponent:incorrectType',...
                                             ['Property ''HiddenActFcn'' shall be chosen among:\n',...
                                              ' - ''linear'',\n',...
                                              ' - ''logistic'',\n',...
                                              ' - ''ReLU'',\n',...
                                              ' - ''tanh'',\n',...
                                              ' - ''softmax''.']);
                        end
                        
                    % Training algorithm
                    case lower('TrainingAlgorithm')
                        
                        switch lower(Values{n}) %#ok<PROPLC>
                            case lower({'Gradient descent', 'GD'})
                            otherwise, error('MyComponent:incorrectType',...
                                            ['Property ''TrainingAlgorithm'' shall be chosen among:\n',...
                                             ' - ''Gradient descent''        (or ''GD''),\n']);
                        end
                    
                    % Bayesian regularization
                    case lower('BayesianRegularization')
                        switch class(Values{n}) %#ok<PROPLC>
                            case 'char'
                                switch lower(Values{n}) %#ok<PROPLC>
                                    case 'on',  Object.BayesianRegularization = true;
                                    case 'off', Object.BayesianRegularization = false;
                                    otherwise, error('MyComponent:incorrectType',...
                                                    ['Property ''BayesianRegularization'' shall be chosen among:\n',...
                                                     ' - false (or ''on''),\n',...
                                                     ' - true  (or ''off'')']);
                                end
                            case 'logical'
                            otherwise, error('MyComponent:incorrectType',...
                                            ['Property ''BayesianRegularization'' shall be chosen among:\n',...
                                             ' - false (or ''on''),\n',...
                                             ' - true  (or ''off'')']);
                        end
                        
                end
                
            end
            
        end
        
        % Method 'get'
        function Value = get(varargin)
            
            if nargin == 1
                disp(varargin{1});
                return
            end
            
            Object = varargin{1};
            Property = varargin{2};
                        
            [Indicator, Property] = Object.isprop(Property);
            
            if Indicator
                Value = Object.(Property);
            else
                error('No appropriate property ''%s'' for class ''MultiLayerPerceptron''.',Property);
            end
            
        end
        
        % Method 'initialize'
        function Object = initialize(Object)
            
            % Default update function
            if isempty(Object.UpdateFcn)
                Object.UpdateFcn = @(MLP)updateFigure(MLP);
            end
            
            % Control of the lengths of layers
            if isempty(Object.LengthsOfLayers)
               error('Parameter ''LengthsOfLayers'' undefined.');
            end
            
            % Control of the consistance between the two last layers
            switch lower(Object.OutputActFcn)
                case 'softmax'
                    if ne(Object.LengthsOfLayers(end-1),Object.LengthsOfLayers(end))
                       error(['Parameter ''LengthsOfLayers'' incompatible with "Softmax" function.\n',...
                              'The lengths of the two last layers shall be equal (here: %u and %u).'],...
                              Object.LengthsOfLayers(end-1),Object.LengthsOfLayers(end)); 
                    end
            end
            
            % Number of layers (input, hidden and output layers)
            Object.NumberOfLayers = numel(Object.LengthsOfLayers);
                        
            % Random initialization (weights and biases)
            for n = 2:Object.NumberOfLayers
                Object.Weights{n}   = (rand(Object.LengthsOfLayers(n),Object.LengthsOfLayers(n-1))-0.5)/2; 
                Object.Biases{n}    = (rand(Object.LengthsOfLayers(n),1)-0.5)/2;
                Object.Values{n}    = NaN(Object.LengthsOfLayers(n),1);                
            end      
            Object.PreviousWeights = Object.Weights;
            Object.PreviousBiases  = Object.Biases;
            
            % Activation functions and derivatives            
            FieldsActFcn              = {'HiddenActFcn',             'OutputActFcn'};
            FieldsActivationFunctions = {'HiddenActivationFunction', 'OutputActivationFunction'};
            FieldsDerivatives         = {'HiddenDerivative',         'OutputDerivative'}; 
            for l = 1:2
                switch lower(Object.(FieldsActFcn{l}))
                    case 'linear'
                        Object.(FieldsActivationFunctions{l}) = @(a) a;
                        Object.(FieldsDerivatives{l})         = @(y) ones(numel(y),1);
                    case 'logistic'
                        Object.(FieldsActivationFunctions{l}) = @(a) 1./(1+exp(-a));
                        Object.(FieldsDerivatives{l})         = @(y) y.*(1-y);
                    case 'relu'             
                        Object.(FieldsActivationFunctions{l}) = @(a) max(a,0);
                        Object.(FieldsDerivatives{l})         = @(y) 1*gt(y,0);
                    case 'softmax'
                        Object.(FieldsActivationFunctions{l}) = @(a) exp(a)/sum(exp(a));
                        Object.(FieldsDerivatives{l})         = @(y) y.*(1-y);
                    case 'tanh'
                        Object.(FieldsActivationFunctions{l}) = @tanh;
                        Object.(FieldsDerivatives{l})         = @(y) 1-y.^2;
                end                
            end
            
        end
        
        % Method 'propagate'
        function Object = propagate(Object,Inputs)
            
            % Initialization
            if eq(Object.NumberOfLayers,0)
               Object = Object.initialize();
            end
                        
            % Preallocation
            Object.Outputs = NaN(Object.LengthsOfLayers(end),size(Inputs,2));
            
            % Outputs calculation
            for i = 1:size(Inputs,2)
                
                % Inputs (first values)
                Object.Values{1} = Inputs(:,i);
                
                % Propagation in hidden layers
                for n = 2:Object.NumberOfLayers-1
                    Object.Values{n} = ...
                        feval(Object.HiddenActivationFunction,...
                              Object.Weights{n}*Object.Values{n-1}+Object.Biases{n});
                end
                
                % Propagation toward outputs
                Object.Values{Object.NumberOfLayers} = ...
                    feval(Object.OutputActivationFunction,...
                          Object.Weights{Object.NumberOfLayers}*Object.Values{Object.NumberOfLayers-1}+Object.Biases{Object.NumberOfLayers});
                
                % Outputs
                Object.Outputs(:,i) = Object.Values{end};
                
            end
            
        end
        
        % Method 'train'
        function Object = train(Object,Inputs,Outputs,Options)
            
            % Options
            if nargin == 4
                Properties = fieldnames(Options);
                Values = struct2cell(Options); %#ok<PROPLC>
                for n = 1:numel(Properties)
                    set(Object,Properties{n},Values{n}); %#ok<PROPLC>
                end
            end
            
            % Initialization
            if eq(Object.NumberOfLayers,0)
               Object = Object.initialize();
            end
            
            % Training
            switch lower(Object.TrainingAlgorithm)   
                case lower({'Gradient descent',   'GD'}), Object.trainGD(Inputs,Outputs);
            end
            
        end
        
    end
    
    % Methods (private)
    methods (Access = 'private')
        
        % Method 'isprop'
        function [Indicator, Property] = isprop(Object,Property)
            
            Properties = properties(Object);
            [Indicator, Indice] = ismember(lower(Property),lower(Properties));
            if Indicator
                Property = Properties{Indice};
            end
            
        end
        
        % Method 'ismethod'
        function [Indicator, Method] = ismethod(Object,Method)
            
            Methods = methods(Object);
            [Indicator, Indice] = ismember(lower(Method),lower(Methods));
            if Indicator
                Method = Methods{Indice};
            end
            
        end
        
        % Method 'splitSets'
        function [Nt,It,Ot,Nv,Iv,Ov] = splitSets(Object,Inputs,Outputs)
            
            % Total number of input vectors
            N = size(Inputs,2);
            
            % Random indices
            Indices = randperm(N);
            
            % Number of inputs for training
            Nt = floor(Object.SplitRatio*N);
            
            % Number of inputs for validation
            Nv = N-Nt;
            
            % Training inputs & outputs
            TrainingIndices = Indices(1:Nt);
            It  = Inputs(:,TrainingIndices);
            Ot = Outputs(:,TrainingIndices);
            
            % Validation inputs & outputs
            ValidationIndices = Indices(Nt+1:end);
            Iv  = Inputs(:,ValidationIndices);
            Ov = Outputs(:,ValidationIndices);
            
        end
        
        % Method 'trainGD', gradient descent algorithm
        function Object = trainGD(Object,Inputs,Outputs)
                        
            % Split between training and validation data
            [Nt, It, Ot, Nv, Iv, Ov] = splitSets(Object,Inputs,Outputs);
            
            % Initialization of cumulation of errors
            if gt(Object.SizeOfBatches,1)
                Object.calculateGradients('zeroize');
                Object.SizeOfBatches = min(Object.SizeOfBatches,Nt);
            end
            
            % Initialization
            e = 0;
            MSE = inf;
            
            % Initialization of figure
            Object.TrainingStep = 'start';
            feval(Object.UpdateFcn,Object);   
            Object.TrainingStep = 'update';
            
            % Epochs
            while lt(e,  Object.NumberOfEpochs) && ...
                  gt(MSE,Object.MinimumMSE)
                
                % Increment of the number of epochs
                e = e+1;
                
                % Mean square error initialization
                MSE = 0;
                
                % Indices random permutation
                Indices = randperm(Nt);
                
                % Propagation of inputs and backpropagation of errors
                for n = 1:Nt
                    
                    % Current index
                    Index = Indices(n);
                    
                    % Propagation of the current input
                    Object.propagate(It(:,Index));
                    
                    % Errors
                    E = Ot(:,Index) - Object.Outputs;    
                    
                    % Back propagation of the current errors                 
                    Object.backPropagateErrors(E);
                    
                    % Weights/biases update
                    switch Object.SizeOfBatches
                        
                        % Stochastic gradient descent
                        case 1                        
                            
                            % Update of weights/biases with the current errors                    
                            Object.updateWeightsWithErrors();
                            
                        % Batch gradient descent
                        otherwise
                            
                            % Cumulation of gradients
                            Object.calculateGradients('cumulate');
                            
                            % Current batch gradient descent
                            if ~mod(n,Object.SizeOfBatches) || eq(n,Nt)
                                
                                % Size of the current batch
                                if ~eq(n,Nt)
                                    Nb = Object.SizeOfBatches;
                                else
                                    Nb = rem(Nt-1,Object.SizeOfBatches)+1;
                                end
                                
                                % Application of mean gradients and zeroize
                                Object.calculateGradients('apply',Nb);
                                
                            end
                            
                    end
                    
                    % Mean square errors
                    MSE = MSE+E'*E;                 
                    
                end
                
                % Current epoch
                Object.CurrentEpoch = e;
                
                % Mean square error for training
                MSE = MSE/2/Nt;
                Object.MeanSquareErrors(e) = MSE;
                
                % Mean square error for validation
                Object.propagate(Iv);                
                E = Object.Outputs-Ov;
                Object.ValidationMeanSquareErrors(e) = sum(sum(E.^2))/2/Nv;                
                
                % Update of figure
                if ~feval(Object.UpdateFcn,Object)
                    break
                end
                
            end
            
            % Update of figure
            Object.TrainingStep = 'stop';
            feval(Object.UpdateFcn,Object);
            
        end
        
        % Method 'backPropagateErrors'
        function Object = backPropagateErrors(Object,Errors)
            
            % Output layer errors
            Object.Errors{Object.NumberOfLayers} = Object.OutputDerivative(Object.Values{Object.NumberOfLayers}) .* Errors;
            
            % Errors propagation
            for n = Object.NumberOfLayers-1:-1:2
                Object.Errors{n} = ...
                    Object.HiddenDerivative(Object.Values{n}) .* ...
                    Object.Weights{n+1}' * Object.Errors{n+1};
            end
            
        end
        
        % Method 'updateWeightsWithErrors'
        function Object = updateWeightsWithErrors(Object)
            
            % Weights update
            for n = 2:Object.NumberOfLayers
                
                for i = 1:Object.LengthsOfLayers(n)
                    
                    for j = 1:Object.LengthsOfLayers(n-1)
                        
                        % Weights
                        Object.Weights{n}(i,j) = ...
                            Object.Weights{n}(i,j) + ...
                            Object.LearningRate * Object.Errors{n}(i) .*  Object.Values{n-1}(j) + ...
                            Object.Momentum * (Object.Weights{n}(i,j) - Object.PreviousWeights{n}(i,j));
                        
                    end
                    
                end
                
                % Previous weights
                Object.PreviousWeights{n} = Object.Weights{n};
                
                % Biases
                Object.Biases{n} = ...
                    Object.Biases{n} + ...
                    Object.LearningRate * Object.Errors{n} + ...
                    Object.Momentum * (Object.Biases{n} - Object.PreviousBiases{n});
                
                % Previous biases
                Object.PreviousBiases{n} = Object.Biases{n};
                
            end
            
        end
        
        % Method 'calculateGradients', 'zeroize', 'cumulate' and 'apply' functions
        function Object = calculateGradients(Object,Action,Value)
            
            switch Action
                
                case 'zeroize'
                    
                    % Zeroization of gradients cumulations
                    for l = 2:Object.NumberOfLayers
                        Object.GradientsWeights{l} = zeros(Object.LengthsOfLayers(l),Object.LengthsOfLayers(l-1));
                        Object.GradientsBiases{l}  = zeros(Object.LengthsOfLayers(l),1);
                    end
                    
                case 'cumulate'
                    
                    % Weights/biases cumulations
                    for n = 2:Object.NumberOfLayers
                        
                        for i = 1:Object.LengthsOfLayers(n)
                            
                            for j = 1:Object.LengthsOfLayers(n-1)
                                
                                % Weights
                                Object.GradientsWeights{n}(i,j) = ...
                                    Object.GradientsWeights{n}(i,j) + ...
                                    Object.Errors{n}(i) .*  Object.Values{n-1}(j);
                                
                            end
                            
                        end
                        
                        % Biases
                        Object.GradientsBiases{n} = ...
                            Object.GradientsBiases{n} + Object.Errors{n};
                        
                    end
                    
                case 'apply'
                                                            
                    % Weights/biases update with cumulations means
                    for n = 2:Object.NumberOfLayers
                        
                        for i = 1:Object.LengthsOfLayers(n)
                            
                            for j = 1:Object.LengthsOfLayers(n-1)
                                
                                % Weights
                                Object.Weights{n}(i,j) = ...
                                    Object.Weights{n}(i,j) + ...
                                    Object.LearningRate * Object.GradientsWeights{n}(i,j)/Value;
                            end
                            
                        end
                        
                        % Biases
                        Object.Biases{n} = ...
                            Object.Biases{n} + ...
                            Object.LearningRate * Object.GradientsBiases{n}/Value;
                                            
                    end
                    
                    % Zeroization
                    Object = calculateGradients(Object,'zeroize');
                    
            end            
            
        end
        
        % Method 'calculateJacobianMatrix'
        function Object = calculateJacobianMatrix(Object,I,N,F)
            
            % Initial weights/biases index
            w = 0;
            
            % Layers
            for l = 2:Object.NumberOfLayers
                
                % Neurons of the current layer
                for n = 1:Object.LengthsOfLayers(l)
                    
                    % Weights of the current neuron
                    for m = 1:Object.LengthsOfLayers(l-1)
                        w = w+1;
                        Object.WeightsVector(w) = Object.Weights{l}(n,m);
                        Object.Weights{l}(n,m) = Object.WeightsVector(w)+Object.dw;
                        Object.propagate(I);
                        for i = 1:N
                            for o = 1:Object.LengthsOfLayers(end)
                                Object.JacobianMatrix((i-1)*Object.LengthsOfLayers(end)+o,w) = (Object.Outputs(o,i)-F(o,i))/Object.dw;
                            end
                        end
                        Object.Weights{l}(n,m) = Object.WeightsVector(w);
                    end
                    
                    % Bias of the current neuron
                    w = w+1;
                    Object.WeightsVector(w) =  Object.Biases{l}(n);
                    Object.Biases{l}(n) = Object.WeightsVector(w)+Object.dw;
                    Object.propagate(I);
                    for i = 1:N
                        for o = 1:Object.LengthsOfLayers(end)
                            Object.JacobianMatrix((i-1)*Object.LengthsOfLayers(end)+o,w) = (Object.Outputs(o,i)-F(o,i))/Object.dw;
                        end
                    end
                    Object.Biases{l}(n) = Object.WeightsVector(w);
                    
                end
                
            end
            
        end
        
        % Method 'updateWeights'
        function Object = updateWeights(Object,W)
            
            w = 0;
            for l = 2:Object.NumberOfLayers
                for n = 1:Object.LengthsOfLayers(l)
                    for m = 1:Object.LengthsOfLayers(l-1)
                        w = w+1;
                        Object.Weights{l}(n,m) = W(w);
                    end
                    w = w+1;
                    Object.Biases{l}(n) = W(w);
                end
            end
            
        end
        
        % Method 'updateFigure'
        function Continue = updateFigure(Object)
            
            persistent Elements
            
            switch Object.TrainingStep
                
                % Creation of figure
                case 'start'
                    Elements = CreationFigure();
                    
              	% Update of figure
                case 'update'
                    switch Object.CurrentEpoch
                        case 1
                            Elements = CreationPlots(Elements,Object);
                        otherwise
                            UpdateFigure(Elements,Object);
                    end
                    
             	% Cancellation of weights/biases update
                case 'cancellation'
                    
              	% Update of button
                case 'stop'
                    if isvalid(Elements.Button)
                        set(Elements.Button,'String','Close');
                    end
                    
            end
            
            % Refreshment
            drawnow();
            
            % Continuation boolean
            Continue = isvalid(Elements.Figure);
            
            % Function 'CreationFigure'
            function Elements = CreationFigure()
                
                % Creation of figure
                SS = get(0,'Screensize');
                W = 400;
                H = 400;
                Elements.Figure = ...
                    figure('Color',       'w',...
                           'NumberTitle', 'Off',...
                           'Resize',      'Off',...
                           'Name',        'Control figure',...
                           'MenuBar',     'None',...
                           'ToolBar',     'None',...
                           'Position',    [SS(3)-W SS(4)-H-30 W H]);
                
                % Creation of stop button
                Elements.Button = ...
                    uicontrol('Style',    'Pushbutton',...
                              'String',   'Stop',...
                              'Position', [W-50 2 50 20],...
                              'Callback', @(~,~)close(Elements.Figure));
                
                % Creation of axes
                Elements.Axes = ...
                    axes('FontSize', 8,...
                         'Box',      'on',...
                         'Xgrid',    'on',...
                         'Ygrid',    'on');
                switch lower(Object.TrainingAlgorithm)
                    case lower({'GD','Gradient descent'})
                        Title = 'Training algorithm: gradient descent';              
                end
                title(Title,'FontWeight','Light','FontSize',8);
                xlabel('Epoch','FontSize',8);
                ylabel('Mean square error',  'FontSize',8);
                hold('on');
                
            end
            
            % Function 'CreationPlots'
            function Elements = CreationPlots(Elements,Object)
                
                Elements.Plots(1) = plot(Object.CurrentEpoch,Object.MeanSquareErrors(Object.CurrentEpoch),'b+-');
                Elements.Plots(2) = plot(Object.CurrentEpoch,Object.ValidationMeanSquareErrors(Object.CurrentEpoch),'c+-');
                Elements.Lines(1) = plot([NaN NaN],[NaN NaN],'g-');
                Elements.Lines(2) = plot([NaN NaN],[NaN NaN],'r-');
                legend(Elements.Plots,{'Training','Validation'},'Location','SouthWest','FontSize',8);
                
            end
            
            % Function 'UpdateFigure'
            function UpdateFigure(Elements,Object)
                
                % Maximum error
                Maximum = max(max(Object.MeanSquareErrors),max(Object.ValidationMeanSquareErrors));
                
                % Update of plots
                set(Elements.Plots(1),...
                    'Xdata',[get(Elements.Plots(1),'Xdata') Object.CurrentEpoch],...
                    'Ydata',[get(Elements.Plots(1),'Ydata') Object.MeanSquareErrors(Object.CurrentEpoch)]);
                set(Elements.Plots(2),...
                    'Xdata',[get(Elements.Plots(2),'Xdata') Object.CurrentEpoch],...
                    'Ydata',[get(Elements.Plots(2),'Ydata') Object.ValidationMeanSquareErrors(Object.CurrentEpoch)]);
                set(Elements.Axes,'Ylim',[0 ceil(Maximum*100)/100]);
                
                % Update of markers
                if gt(Object.CurrentEpoch,50)
                    set(Elements.Plots,'Marker','None');
                end
                
                % Update of abscissa axes
                Xlim = get(Elements.Axes,'Xtick');
                if ge(Xlim(end),Object.NumberOfEpochs)
                    set(Elements.Lines(2),...
                        'Xdata',[Object.NumberOfEpochs Object.NumberOfEpochs],...
                        'Ydata',[Object.MinimumMSE Maximum]);
                end
                
                % Update of ordinates axes
                if Object.MeanSquareErrors(Object.CurrentEpoch) > 0.05
                    Ylim = get(Elements.Axes,'Ytick');
                    set(Elements.Axes,'Ylim',[0 Ylim(end)+Ylim(2)]);
                else
                    set(Elements.Axes,'Ylimmode','auto','Yscale','log');
                end
                Ylim = get(Elements.Axes,'Ytick');
                if le(Ylim(1),Object.MinimumMSE)
                    set(Elements.Lines(1),...
                        'Xdata',[1 min(Xlim(end),Object.NumberOfEpochs)],...
                        'Ydata',[Object.MinimumMSE Object.MinimumMSE]);
                end
                
            end
            
        end
        
    end

end
