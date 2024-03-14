%% Bulls eye/cross air fixation target
function time_out=bullsEyeFixation(win,time,img_duration,fixFlipTime,rect,width,distScreen, ...
    cx,cy,colorOval,colorCross,d1,d2)
% width = 50; % horizontal dimension of display (cm)
% dist = 60; % viewing distance (cm)

% colorOval = [255 0 0]; % color of the two circles [R G B]
% colorCross = [0 0 0]; % color of the Cross [R G B]

% d1 = 2; % diameter of outer circle (degrees)
% d2 = 0.2; % diameter of inner circle (degrees)

ppd=pi*(rect(3)-rect(1))/atan(width/distScreen/2)/360; % pixel per degree

Screen('FillOval',win,colorOval,[cx-d1/2*ppd,cy-d1/2*ppd,cx+d1/2*ppd,cy+d1/2*ppd],d1*ppd);
Screen('DrawLine',win,colorCross,cx-d1/2*ppd,cy,cx+d1/2*ppd,cy,d2*ppd);
Screen('DrawLine',win,colorCross,cx,cy-d1/2*ppd,cx,cy+d1/2*ppd,d2*ppd);
Screen('FillOval',win,colorOval,[cx-d2/2*ppd,cy-d2/2*ppd,cx+d2/2*ppd,cy+d2/2*ppd],d2*ppd);
time_out=Screen('Flip',win,time+img_duration-fixFlipTime);
end
