%% EEG encoding - resting state (paradigm 3)
clear
clc

KbName('UnifyKeyNames') % (following MacOS-X naming scheme)


%% 1. To edit
% Triggers for EEG recordings
cfg.eeg_mode=1;

% Resting state duration
p.resting_state_duration=300; % 5 minutes (300 seconds)

% Data directories
eegDir='D:\cichyLab\#Common\parallel_port';

% Screen window number for PTB 'OpenWindow' function
screenWin=0;
distFromScreen=600; % distance from screen, in millimeters

% Milliseconds to be subtracted from the flipping time so to not miss the
% visual frame (and having to wait an additional 16ms (or more, or less,
% depending on the display device refresh rate) for the flip).
% If not needed, set to 0
fixFlipTime=.01;

% Input device number for PTB keyboard functions
deviceNum=-1;

% Name of response keys (following MacOS-X naming scheme)
escButton='ESCAPE'; % escape button, to be pressed (during the response period) to exit the experiment


%% 2. Triggers for EEG recordings
if cfg.eeg_mode==1
    addpath(eegDir);
    cfg.trigger_delay_time=0.011; % screen delay (half a flip + 3ms)
end


%% 3. Opening Psychtoolbox and Instructions
[win, screenRect]=Screen('OpenWindow',screenWin,[159 162 166],[]);
Screen('BlendFunction',win,'GL_SRC_ALPHA', 'GL_ONE_MINUS_SRC_ALPHA');

HideCursor(0)
Screen('TextSize',win,floor(screenRect(3)/50));
Screen('TextStyle', win,0);

% Instructions
ListenChar(2);
while true
    instructions='RESTING STATE\n\n\nPlease fixate the central dot.\n\nDon''t move your face and body.\n\nTry to blink as little as possible.\n\n';
    DrawFormattedText(win,instructions,'center','center',[0 0 0]);
	bullsEyeFixation_2(win,0,screenRect,30,distFromScreen/10, ...
			screenRect(3)/2,screenRect(4)/2,[255 0 0],[0 0 0],.4,.08);
    Screen('Flip',win);
    [keyIsDown, secs, keyCode]=KbCheck(deviceNum);
    if keyIsDown
        break
    end
end


%% 11. Resting state and triggers
% Test and target trials only have 1 trigger 
% Onset: (1)
% Offset: (2)

 % Blank screen with fixation dot
bullsEyeFixation_2(win,1,screenRect,30,distFromScreen/10, ...
		screenRect(3)/2,screenRect(4)/2,[255 0 0],[0 0 0],.4,.08);

% Onset trigger
if cfg.eeg_mode==1
	WaitSecs(cfg.trigger_delay_time); % Wait Xms before sending the trigger
	send_triggerIO64(1);
end

% Resting state duration
time=GetSecs;
while true
	[secs, keyCode]=KbPressWait(deviceNum,time+p.resting_state_duration);
	thisResp=KbName(keyCode);
	if strcmp(thisResp,escButton) || isempty(thisResp)
		break
	end
end

% Offset trigger
if cfg.eeg_mode==1
	WaitSecs(cfg.trigger_delay_time); % Wait Xms before sending the trigger
	send_triggerIO64(2);
end

sca
ListenChar(0);

