%% Example: classification of sectors with gradient descent algorithm
function MLP = ExampleClassificationSectors()

close('all','force'); clc;

% Parameters
N = 24;
Nt = N^2;

% Regular coordinates
V = linspace(-2,+2,N);       
[X,Y] = ndgrid(V,V);      
Coordinates = [X(:)';Y(:)'];   

% Regions
Regions = 1*ge(Y,X)+2*(gt(Y,-X)|lt(abs(X+Y),1e-2))+1;
Regions = Regions(:)';

% Probabilities
Probabilities = zeros(4,numel(Regions));
for n = 1:numel(Regions)
    Probabilities(Regions(n),n) = 1;
end

% Figure
Figure = figure('Color','w');
axis('off');
hold('on');
Colors = {'m','g','c','y'};

% Full screen
jFrame = get(Figure,'JavaFrame');
drawnow(); pause(0.1);
jFrame.setMaximized(true);

% Plot of the coordinates
for r = 1:4
    I = find(eq(Regions,r));
    plot(Coordinates(1,I),Coordinates(2,I),'o',...
        'MarkerEdgeColor',  'k',...
        'MarkerFaceColor',  Colors{r},...
        'MarkerSize',       6);
end        
xlim([-2 +2]);
ylim([-2 +2]);
axis('equal');
drawnow();

% Coordinates of map components
N2 = 300;
V = linspace(-2,+2,N2);       
[X,Y] = ndgrid(V,V);      
Coordinates2 = [X(:)';Y(:)']; 

% Map color
Colors = ...
    [1 0 0;...
     0 1 0;...
     0 0 1;...
     1 1 0];

% Multi-layer perceptron
MLP = ...
    MultiLayerPerceptron('LengthsOfLayers', [2 4 4],...
                         'HiddenActFcn',    'linear',...
                         'OutputActFcn',    'softmax',...
                         'UpdateFcn',       'default');           

% Training options
Options = ...
    struct('TrainingAlgorithm',         'GD',...
           'NumberOfEpochs',            100,...
           'MinimumMSE',                1e-2,...
           'SizeOfBatches',             10,...
           'SplitRatio',                1,...
           'Momentum',                  0.9,...
           'UpdateFcn',                 @Update);                    
       
% Training       
MLP.train(Coordinates,Probabilities,Options);     
    
    % Figure update function
    function Continue = Update(MLP)
        
        persistent Coloration Epoch VR
        
        % Training continuation boolean
        Continue = true;
        
        switch MLP.TrainingStep
            
            case 'start'
                
                % Creation of the video
                Name = 'Sectors';
                VR = ...
                    VideoRecorder('Filename',     [Name datestr(now,'_dd-mm-yy_HH-MM-SS')],...
                                  'Fileformat',   'MPEG-4',...
                                  'Figure',       Figure);
                
                % Update of the video
                for i = 1:20
                    VR.add();
                end
                
                return
                
            case 'cancellation'
                
                % Cancellation of the current training step
                return
                
            case 'Update'
                
                % Update of the current step except if the epoch is the same
                if MLP.CurrentEpoch == Epoch
                    return
                end
                
            case 'stop'
                
                % Ending of the video
                VR.stop();
                
                return
                
        end
        
        % Number of erroneous regions
        MLP.propagate(Coordinates);
        [~,R] = max(MLP.Outputs);
        E = sum(~eq(R,Regions));        
        if eq(E,0)
            Continue = false; 
        end
        
        % Title update
        title(sprintf('Epoch: %02u, Errors: %u/%u',MLP.CurrentEpoch,E,Nt));
        
        % Most probable regions     
        MLP.propagate(Coordinates2);                
        Probabilities2 = MLP.Outputs;                   
        [~, Regions2] = max(Probabilities2,[],1);
        
        switch MLP.CurrentEpoch
            
            case 1
            
                % Creation of the map
                Coloration = ...
                    pcolor(reshape(Coordinates2(1,:),N2,N2),...
                           reshape(Coordinates2(2,:),N2,N2),...
                           reshape(Regions2,N2,N2));
                set(Coloration,...
                    'EdgeColor', 'none',...
                    'FaceAlpha', 0.25);        
                colormap(Colors);
                uistack(Coloration,'bottom');
            
            otherwise
                
                % Update of the map
                set(Coloration,...
                    'Xdata', reshape(Coordinates2(1,:),N2,N2),...
                    'Ydata', reshape(Coordinates2(2,:),N2,N2),...
                    'Cdata', reshape(Regions2,N2,N2));                      
                colormap(Colors);
                
        end
        
        % Update of the video
        drawnow();  
        for i = 1:20
            VR.add();
        end
        
        % Current epoch
        Epoch = MLP.CurrentEpoch;
                
    end

end
