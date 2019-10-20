%% Video Recorder
%
%  VERSIONS
%  - v1.0  (21th mar. 2016) : initial version
%  - v1.01 (26th mar. 2016) : correction of a bug in record file naming
%
%  OBJECT METHODS :
%  - VR = VideoRecorder(PROPERTY1,VALUE1,PROPERTY2,VALUE2,...) : Create a Video Recorder object and set its values (if defined)
%  - VR.set(PROPERTY1,VALUE1,PROPERTY2,VALUE2,...)             : Set value(s)
%  - VR.get(PROPERTY)                                          : Get value(s) (do not define PROPERTY argument to display all properties)
%  - VR.start()                                                : Start a timer recording a frame every PERIOD
%  - VR.stop()                                                 : Stop video record (and the timer potentially started through 'start' method)
%  - VR.add()                                                  : Add a frame to video without using a timer
%  - VR.delete()                                               : Delete object components (VideoWriter and timer)
%
%  OBJECT PROPERTIES :
%  - FILENAME           : String specifying the type of file to write.
%  - FILEFORMAT         : - 'Archival'         : Motion JPEG 2000 file with lossless compression.
%                         - 'Motion JPEG AVI'  : Compressed AVI file using Motion JPEG codec.
%                         - 'Motion JPEG 2000' : Compressed Motion JPEG 2000 file.
%                         - 'MPEG-4'           : Compressed MPEG-4 file with H.264 encoding.
%                         - 'Uncompressed AVI' : Uncompressed AVI file with RGB24 video.
%                         - 'Indexed AVI'      : Uncompressed AVI file with indexed video.
%                         - 'Grayscale AVI'    : Uncompressed AVI file with grayscale video.
%  - PERIOD             : Period of frame recording [s] (if a timer is used).
%  - TASKSTOEXECUTE     : Number of frames to write (if a timer is used).
%  - EXECUTIONMODE      : Mode used to schedule timer events  (if a timer  is used).
%  - FIGURE             : handle of the figure whose frames are recorded.
%  - DYNAMIC PROPERTIES : - class 'VideoWriter' properties (type "help VideoWriter").
%                         - class 'timer' properties except 'StartFcn', 'TimerFcn' and 'StopFcn' (type "help timer").
%  
%  EXAMPLE 1 :
%  
%  % Video file name
%  Filename = 'Video_sinus_comet';
%   
%  % Figure representing sinus function
%  Figure = figure('Color','w');
%  title('Sinus','fontweight','light','fontsize',9);
%  xlabel('\theta [rad]','fontweight','light','fontsize',9);
%  ylabel('sin(\theta)','fontweight','light','fontsize',9);
%  grid('on');
%  box('on');
%  hold('on');
%  
%  % Creation of a Video object
%  VR = VideoRecorder('Filename',     Filename,...
%                     'Fileformat',   'MPEG-4',...
%                     'Figure',       Figure,...
%                     'Period',       0.05,...
%                     'BusyMode',     'error');
%  
%  % Saving start                
%  VR.start();
%  
%  % Progressive representation of sinus function
%  N = 500;
%  k = 5;
%  X = linspace(0,k*2*pi,N);
%  Y = sin(X);
%  comet(X,Y);
%  
%  % Saving stop
%  VR.stop();
%  
%  % Video_recorder reading
%  Filename = get(VR,'Filename');
%  try
%      winopen(Filename);
%  catch ME
%      error('Video_recorder "%s" cannot be read (%s).',Filenamename,ME.message);
%  end
%
%  EXAMPLE 2 :
%
%  % Video file name
%  Filename = 'Video_sinus_plot';
%  
%  % Figure representing sinus function
%  Figure = figure('Color','w');
%  title('Sinus','fontweight','light','fontsize',9);
%  xlabel('\theta [rad]','fontweight','light','fontsize',9);
%  ylabel('sin(\theta)','fontweight','light','fontsize',9);
%  grid('on');
%  box('on');
%  hold('on');
%  
%  % Creation of a Video object
%  VR = VideoRecorder('Filename',     Filename,...
%                     'Figure',       Figure);
%  
%  % Progressive representation of sinus function
%  N = 500;
%  k = 5;
%  X = linspace(0,k*2*pi,N);
%  Y = sin(X);
%  
%  h = plot(0,0);
%  
%  for n = 1:N
%       I = 1:n;
%       set(h,...
%           'Xdata',X(I),...
%           'Ydata',Y(I));
%       VR.add();
%  end
%  
%  % Saving stop
%  VR.stop();
%  
%  % Video reading
%  Filenamename = get(VR,'Filename');
%  try
%      winopen(Filenamename);
%  catch ME
%      error('Video "%s" cannot be read (%s).',Filenamename,ME.message);
%  end

classdef VideoRecorder < hgsetget & dynamicprops
    
    % Properties (public access)
    properties (Access = 'public')
        Filename       = 'Default';
        FileFormat     = 'MPEG-4';
        Figure         = [];
        Period         = 0.05;
        TasksToExecute = inf;
        ExecutionMode  = 'fixedRate';
    end
    
    % Properties (private access)
    properties (Access = 'private')
        IsOpen   = false;
        Timer    = timer();      
        Recorder = VideoWriter('default');
    end
    
    % Methods
    methods
        
        % Constructor
        function Object = VideoRecorder(varargin)
            
            for v = 1:2:length(varargin)
                
                Property = varargin{v};
                Value = varargin{v+1}; 
                set(Object,Property,Value);
                
            end
            
        end
        
        % Function 'set'
        function Object = set(Object,varargin)
            
            Properties = varargin(1:2:end);
            Values = varargin(2:2:end);
            
            for n = 1:length(Properties)
                
                [is1, Property1] = isproperty(Object,Properties{n});                
                [is2, Property2] = ispropertyvideowriter(Object,Properties{n});   
                [is3, Property3] = ispropertytimer(Object,Properties{n});
                
                if is1
                    Object.(Property1) = Values{n};
                end
                if is2
                    switch Property2
                        case {'Filename','FileFormat'}
                        otherwise
                            set(Object.Recorder,Property2,Values{n});
                    end
                end
                if is3
                    switch Property3
                        case {'StartFcn','TimerFcn','StopFcn'}
                            warning('Property "%s" is reserved !',Property3);                            
                        otherwise
                            set(Object.Timer,Property3,Values{n});
                    end
                end
                if ~is1 && ~is2 && ~is3
                    error('Property "%s" not supported !',Properties{n});
                end
                
            end          
            
            % Creation of timer and recorder
            Object = create_timer_recorder(Object);
            
        end
        
        % Creation of timer and recorder
        function Object = create_timer_recorder(Object)
                
            Object.Timer = ...
                timer('StartFcn',       @(~,~)Object.TimerStart,...
                      'TimerFcn',       @(~,~)Object.TimerRecord,...  
                      'StopFcn',        @(~,~)Object.TimerStop,...
                      'Period',         Object.Period,...
                      'TasksToExecute', Object.TasksToExecute,...
                      'ExecutionMode',  Object.ExecutionMode);
            
            Object.Recorder = VideoWriter(Object.Filename,Object.FileFormat);
            
        end
        
        % Function 'get'
        function Value = get(varargin)
            
            switch nargin
                
                case 1
                    
                    % Video recorder
                    Object = varargin{1};
                   
                    % Video writer
                    Properties = fieldnames(VideoWriter('default'));
                    for p = 1:numel(Properties)
                        if ~isprop(Object,Properties{p})
                            addprop(Object,Properties{p});
                        end
                        Object.(Properties{p}) = Object.Recorder.(Properties{p});
                    end
                    
                    % Timer
                    Properties = fieldnames(timer);
                    for p = 1:numel(Properties)                        
                        if ~isprop(Object,Properties{p})                            
                            switch Properties{p}
                                case {'StartFcn','TimerFcn','StopFcn'}
                                    warning('Property "%s" is reserved !',Properties{p});  
                                otherwise
                                    addprop(Object,Properties{p});                          
                                    Object.(Properties{p}) = Object.Timer.(Properties{p});
                            end
                        end
                    end
                    
                    % Video recorder with dynamic properties
                    Value = Object;
                    
                otherwise
                    
                    Object = varargin{1};
                    Property = varargin{2};
                    [is1, Property1] = isproperty(Object,Property);    
                    [is2, Property2] = ispropertyvideowriter(Object,Property);                    
                    [is3, Property3] = ispropertytimer(Object,Property);
                
                    if is1
                        switch Property1
                            case 'Filename'
                                Value = Object.Recorder.Filename;
                            otherwise
                                Value = Object.(Property1);
                        end
                    elseif is2
                        Value = Object.Recorder.(Property2);                        
                    elseif is3                        
                        switch Property3
                            case {'StartFcn','TimerFcn','StopFcn'}
                                Value = [];
                                warning('Property "%s" is reserved !',Property3);
                            otherwise
                                Value = Object.Timer.(Property3);
                        end
                    else
                        error('Property "%s" not supported !',Property);
                    end
                    
            end
            
        end
        
        % Function 'isproperty'
        function [is, Property] = isproperty(Object,Property)
            
            Properties = fieldnames(Object); 
            [is, b] = ismember(lower(Property),lower(Properties));
            if b
                Property = Properties{b};
            else
                Property = '';
            end
            
        end
        
        % Function 'isproperty' for video writer object
        function [is, Property] = ispropertyvideowriter(Object,Property)
            
            Properties = properties(Object.Recorder);
            [is, b] = ismember(lower(Property),lower(Properties));
            if b
                Property = Properties{b};
            else
                Property = '';
            end
            
        end
        
        % Function 'isproperty' for timer object
        function [is, Property] = ispropertytimer(Object,Property)
            
            Properties = fieldnames(Object.Timer);
            [is, b] = ismember(lower(Property),lower(Properties));
            if b
                Property = Properties{b};
            else
                Property = '';
            end
            
        end  
        
        % Function 'add'
        function Object = add(Object)
            
            if ~Object.IsOpen
                Object.Recorder = VideoWriter(Object.Filename,Object.FileFormat);
                open(Object.Recorder);
                Object.Period = -1;
                Object.IsOpen = true;
            end
                          
            Frame = getframe(Object.Figure);
            writeVideo(Object.Recorder,Frame);
                        
        end
        
        % Function 'start'
        function start(Object)
            
            start(Object.Timer);
            
        end
        
        % Function 'stop'
        function Object = stop(Object)
           
            stop(Object.Timer);
            close(Object.Recorder);
            Object.IsOpen = true;
            
        end
        
        % Function 'delete'
        function Object = delete(Object)
            
            delete(Object.Timer);
            if isfield(Object,'Recorder')
                close(Object.Recorder);
                delete(Object.Recorder);
            end
            Object = [];
            
        end
        
        % Function 'TimerStart'
        function TimerStart(Object)
            
            Object.Recorder = VideoWriter(Object.Filename,Object.FileFormat);
            open(Object.Recorder);
            
                   
        end 
        
        % Function 'TimerRecord'
        function TimerRecord(Object)
                        
            Frame = getframe(Object.Figure);            
            writeVideo(Object.Recorder,Frame);
                        
        end
        
        % Function 'TimerStop'
        function TimerStop(Object)
                        
            close(Object.Recorder);
            
        end
        
    end
    
end
