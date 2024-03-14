%% This function generates a structure with the training and target images
function stimuli=get_stimuli_train(imgDir,p)

%% Training
dir_training=dir([imgDir, '/training_images']);
dir_training=dir_training(3:end);
idx=1;

for c=1:length(dir_training)
    img_dir=dir([dir_training(c).folder, '/', dir_training(c).name]);
    img_dir=img_dir(3:end);

	for n=1:p.img_per_cat %length(img_dir)
		stimuli.training(idx).category_name=dir_training(c).name;
		stimuli.training(idx).num_category=c;
		stimuli.training(idx).num_img=n;
		img=imread([img_dir(n).folder, '/', img_dir(n).name]);
		stimuli.training(idx).image=img;
		idx=idx+1;
	end
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

