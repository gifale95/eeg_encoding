%% EEG encoding - test (paradigm 3)
clear
clc
%PerceptualVBLSyncTest

rng('shuffle'); % randomization of trials according to Matlab internal time
KbName('UnifyKeyNames') % (following MacOS-X naming scheme)

% Get rid of warning messages %%
Screen('Preference', 'SkipSyncTests', 1);
Screen('Preference','VisualDebugLevel', 0);
Screen('Preference','SuppressAllWarnings', 1);

%% 1. To edit
% Triggers for EEG recordings
cfg.eeg_mode=0;

% Data directories
%imgDir='D:\cichyLab\ale\eeg_encoding\paradigm_3\stimuli\px-400\';
%saveDir='D:\cichyLab\ale\eeg_encoding\paradigm_3\collected_data\';
%eegDir='D:\cichyLab\#Common\parallel_port';
imgDir='/home/ale/aaa_stuff/PhD/studies/eeg_encoding/paradigm_3/stimuli/used_images';

% Screen info
distFromScreen=600; % distance from screen, in millimeters
screenWidth=340; % screen width, in millimeters

% Screen window number for PTB 'OpenWindow' function
screenWin=0;

% Milliseconds to be subtracted from the flipping time so to not miss the
% visual frame (and having to wait an additional 16ms (or more, or less,
% depending on the display device refresh rate) for the flip).
% If not needed, set to 0
fixFlipTime=.01;

% Input device number for PTB keyboard functions
deviceNum=-1;

% Name of response keys (following MacOS-X naming scheme)
rightButton='RightArrow'; % button on the rigth, to be pressed when the target is present
leftButton='LeftArrow'; % button on the left, to be pressed when no target is present
escButton='ESCAPE'; % escape button, to be pressed (during the response period) to exit the experiment


%% 2. Triggers for EEG recordings
if cfg.eeg_mode==1
    addpath(eegDir);
    cfg.trigger_delay_time=0.011; % screen delay (half a flip + 3ms)
end


%% 3. Establishing the paradigm
% Timing
p.img_duration=.1; % 100ms of image onscreen duration
p.SOA=.2; % 200ms of SOA
p.pre_response_time=.75; % 750ms between the last sequence image & the blinking/response screen
p.time_for_response=2; % 2s for blink & response
p.intra_sequence_time=.75; % 750ms between response and next sequence

% Presentation structure
p.img_per_sequence=20;
p.sequences_per_block=51;
p.test_blocks=4;
p.targets_per_block=6;

% Stimuli
p.test_cat=200;
p.img_per_cat=1;
p.tot_n_test=p.test_cat*p.img_per_cat;
p.rep_test=20;
p.extra_test_rep=p.img_per_sequence*p.sequences_per_block*p.test_blocks- ...
    p.targets_per_block*p.test_blocks-p.tot_n_test*p.rep_test;

% Images visual angle
p.visAngleX=7; % Horizontal visual angle
p.visAngleY=7; % Vertical visual angle


%% 4. Inserting the paradigm info into the results structure
% Timing
data.paradigm_info.img_duration=p.img_duration;
data.paradigm_info.SOA=p.SOA;
data.paradigm_info.pre_response_time=p.pre_response_time;
data.paradigm_info.blink_and_response_time=p.time_for_response;
data.paradigm_info.intra_sequence_time=p.intra_sequence_time;

% Presentation structure
data.paradigm_info.img_per_sequence=p.img_per_sequence;
data.paradigm_info.sequences_per_block=p.sequences_per_block;
data.paradigm_info.test_blocks=p.test_blocks;
data.paradigm_info.targets_per_block=p.targets_per_block;

% Stimuli
data.paradigm_info.test_categories=p.test_cat;
data.paradigm_info.images_per_cat=p.img_per_cat;
data.paradigm_info.tot_test_img=p.tot_n_test;
data.paradigm_info.test_repetitions=p.rep_test;
data.paradigm_info.extra_test_rep=p.extra_test_rep;
data.paradigm_info.tot_targets=p.targets_per_block*p.test_blocks;

% Images visual angle
data.paradigm_info.visual_angle=p.visAngleX;


%% 5. Asking subject's info
% Info: ID, Session, Age, Sex.
sub_ID=input('Subject''s Number: --> ');
sub_session=input('Subject''s Session: --> ');
sub_age=input('Subject''s Age: --> ');
sub_gender=input('Subject''s Gender: --> ','s');

% Putting the subject's info into the data structure
data.subject_info.ID=sub_ID;
data.subject_info.session=sub_session;
data.subject_info.age=sub_age;
data.subject_info.sex=sub_gender;


%% 6. Reading the stimuli into a structure % defining the presentation order
% This function generates a structure with the training and test stimuli
stimuli=get_stimuli_test(imgDir);

% This function generates the randomized and balanced stimuli order
stimOrder=defineStimuliOrder(p);


%% 7. Opening Psychtoolbox and Instructions
[win, screenRect]=Screen('OpenWindow',screenWin,[159 162 166],[]);
Screen('BlendFunction',win,'GL_SRC_ALPHA', 'GL_ONE_MINUS_SRC_ALPHA');

HideCursor(0)
Screen('TextSize',win,floor(screenRect(3)/50));
Screen('TextStyle', win,0);

% Instructions
ListenChar(2);
while true
    instructions='WELCOME TO THIS EXPERIMENT\n\n\nSequences of images will be presented to you.\n\nYour task is to report whether BUZZ LIGHTYEAR is present in each sequence.\n\nIf you see BUZZ, press the RIGHT ARROW key during the response period.\n\nIf you don''t see BUZZ, press the LEFT ARROW key during the response period.\n\nBe as accurate as possible.\n\n\n\nPress any key to continue';
    DrawFormattedText(win,instructions,'center','center',[0 0 0]);
	bullsEyeFixation_2(win,0,screenRect,30,distFromScreen/10, ...
			screenRect(3)/2,screenRect(4)/2,[255 0 0],[0 0 0],.4,.08);
    Screen('Flip',win);
    [keyIsDown, secs, keyCode]=KbCheck(deviceNum);
    if keyIsDown
        break
    end
end


%% 8. Defining stimuli size
% Stimuli size
[stimSizeX, stimSizeY]=visangle2stimsize(p.visAngleX,p.visAngleY,distFromScreen, ...
    screenWidth,screenRect(3));

% Coordinates of the destination rectangle
destRect=[(screenRect(3)/2)-floor(stimSizeX/2) (screenRect(4)/2)- ...
    floor(stimSizeY/2) (screenRect(3)/2)+floor(stimSizeX/2) ...
    (screenRect(4)/2)+floor(stimSizeY/2)];


%% 9. Task
image=1; % trial counter
response_counter=1;

for block=1:p.test_blocks % blocks
    for sequence=1:p.sequences_per_block % sequences
        
        target_occurrence=0;
        
        % Blank screen with fixation dot
        bullsEyeFixation_2(win,1,screenRect,30,distFromScreen/10, ...
            screenRect(3)/2,screenRect(4)/2,[255 0 0],[0 0 0],.4,.08);
        pause(p.intra_sequence_time)
        
        for img=1:p.img_per_sequence % images
            
            
            %% 10. Image presentation
            % Saving the trial info into the structure
            data.images(image).block=block;
            data.images(image).sequence=sequence;
            data.images(image).img=img;
            
            if stimOrder(img,sequence,block)==0 % target images trials
                idx=randi([1 length(stimuli.target)]);
                tex=Screen('MakeTexture',win,stimuli.target(idx).image);
                target_occurrence=1;
                data.images(image).img_type='target';
                data.images(image).img_category=stimOrder(img,sequence,block);
                data.images(image).within_category_number=stimOrder(img,sequence,block);
                data.images(image).tot_img_number=stimOrder(img,sequence,block);
                data.images(image).category_name=stimOrder(img,sequence,block);
                trigger_1=255; % trigger n
                data.images(image).trigger_1=trigger_1;
                
            else  % test images
                tex=Screen('MakeTexture',win,stimuli.test(stimOrder(img,sequence,block)).image);
                data.images(image).img_type='test';
                data.images(image).img_category=stimuli.test(stimOrder(img,sequence,block)).num_category;
                data.images(image).within_category_number=1;
                data.images(image).tot_img_number=stimOrder(img,sequence,block);
                data.images(image).category_name=stimuli.test(stimOrder(img,sequence,block)).category_name;
                trigger_1=data.images(image).img_category; % trigger n
                data.images(image).trigger_1=trigger_1;
            end

            % Drawing the images and the fixation dot
            Screen('DrawTexture',win,tex,[],destRect);
            bullsEyeFixation_2(win,0,screenRect,30,distFromScreen/10, ...
                screenRect(3)/2,screenRect(4)/2,[255 0 0],[0 0 0],.4,.08);
            
            % Flipping the images and the fixation dot
            if img==1 % if 1st img of the sequence, flip directly
                time=Screen('Flip',win);
            else % if not 1st img of the sequence, flip after SOA time
                time=Screen('Flip',win,time+p.SOA-fixFlipTime);
            end
            
            
            %% 11. Triggers
            % Test and target trials only have 1 trigger 
            % Test: (1-200) --> (Category)
            % Target: (255) --> (Target)
            
            % 1st trigger
            if cfg.eeg_mode==1
                WaitSecs(cfg.trigger_delay_time); % Wait Xms before sending the trigger
                send_triggerIO64(trigger_1);
            end
            
            
            %% 12. Image offset & timing
            % Image screen offset
            time_2=bullsEyeFixation(win,time,p.img_duration,fixFlipTime,screenRect, ...
                30,distFromScreen/10,screenRect(3)/2,screenRect(4)/2,[255 0 0],[0 0 0],.4,.08);
            
            % Saving the single img presention time and SOA into the
            % results structure
            data.images(image).img_duration=time_2-time;
            if img==1 % if 1st image of the sequence no SOA time saved
                data.images(image).SOA=0;
            else % if not the 1st image of the sequence calculate and save SOA time
                data.images(image).SOA=time-img_duration_2;
            end
            img_duration_2=time;
            
            % Closing the texture after each trial (to save computation time)
            Screen('Close',tex)
            image=image+1;
        end % image
        
        
        %% 13. Blinking & response
        % Blinking period
        blink=sprintf('First Blink, then Respond');
        DrawFormattedText(win,blink,'center',screenRect(4)/2.2,[0 0 0]);
		bullsEyeFixation_2(win,0,screenRect,30,distFromScreen/10, ...
				screenRect(3)/2,screenRect(4)/2,[255 0 0],[0 0 0],.4,.08);
        Screen('Flip',win,time+p.pre_response_time-fixFlipTime);
        
        % Collecting responses
        resp_time_1=GetSecs;
        while true
            [secs, keyCode]=KbPressWait(deviceNum,resp_time_1+p.time_for_response);
            thisResp=KbName(keyCode);
            if strcmp(thisResp,leftButton) || strcmp(thisResp,rightButton) || strcmp(thisResp,escButton) || isempty(thisResp)
                break
            end
        end
        
        if strcmp(thisResp,leftButton)
            response=0;
        elseif strcmp(thisResp,rightButton)
            response=1;
        elseif isempty(thisResp)
            response=2; % no response
        elseif strcmp(thisResp,escButton) % Escape key
            Screen('CloseAll')
            ListenChar(0);
        end
        
        if target_occurrence==response
            correctness=1;
        else
            correctness=0;
        end
        
        % Putting the behavioral results into a structure
        data.task(response_counter).block=block;
        data.task(response_counter).sequence=sequence;
        data.task(response_counter).target=target_occurrence;
        data.task(response_counter).response=response;
        data.task(response_counter).correctness=correctness;
        response_counter=response_counter+1;
    end % sequence
    
    
    %% 14. Break
    if block<p.test_blocks
        Screen('Flip',win);
        instructions=sprintf('END OF RUN %d/%d\n\n\nYou can now take a break\n\n\n\nPress any key when you wish to continue',block,p.test_blocks);
        DrawFormattedText(win,instructions,'center','center',[0 0 0]);
		bullsEyeFixation_2(win,0,screenRect,30,distFromScreen/10, ...
				screenRect(3)/2,screenRect(4)/2,[255 0 0],[0 0 0],.4,.08);
        Screen('Flip',win);
        [keyIsDown, secs, keyCode]=KbPressWait(deviceNum);
    end
end % block

sca
ListenChar(0);


%% 15. Saving the data structure
% Save the data structure with the ID name
save([saveDir, '\sub-', num2str(sub_ID), '_sess-', num2str(sub_session),'_data-test'],'data')

