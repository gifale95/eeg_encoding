%% This function generates a structure with the test and target images
function stimuli=get_stimuli_test(imgDir)

%% Test
dir_test=dir([imgDir,'/test_images']);
dir_test=dir_test(3:end);

for c=1:length(dir_test)
    stimuli.test(c).category_name=dir_test(c).name;
    stimuli.test(c).num_category=c;
    img_dir=dir([dir_test(c).folder, '/', dir_test(c).name]);
    img_dir=img_dir(3:end);
    img=imread([img_dir(1).folder, '/', img_dir(1).name]);
    stimuli.test(c).image=img;
end


%% Target
dir_target=dir([imgDir, '/target_images']);
dir_target=dir_target(3:end);

for c=1:length(dir_target)
    stimuli.target(c).file_name=dir_target(c).name;
    stimuli.target(c).img_number=c;
    stimuli.target(c).image=imread([dir_target(c).folder, '/', dir_target(c).name]);
end
end

