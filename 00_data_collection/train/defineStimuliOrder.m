%% Establishing and saving the randomized stimuli order
clear
clc
rng('shuffle')


%% Parameters
% Save dir
saveDir='/home/ale/aaa_stuff/PhD/studies/eeg_encoding/paradigm_3/code/002-eeg_data_collection/train/new/train_order';

% Subjects and sessions
p.sub=10;
p.ses=4;

% Presentation structure
p.img_per_sequence=20;
p.sequences_per_block=56;
p.training_blocks=15;
p.targets_per_block=6;

% Stimuli
p.tot_training_cat=1654;
p.training_cat=p.tot_training_cat/2;
p.img_per_cat=10;
p.tot_n_training=p.training_cat*p.img_per_cat;
p.rep_training=2;
p.extra_train_rep=p.img_per_sequence*p.sequences_per_block*p.training_blocks- ...
    p.targets_per_block*p.training_blocks-p.tot_n_training*p.rep_training;


%% Looping across subjects and sessions
for sub=7:p.sub
	
	% Equally dividing the stimuli into the 4 sessions
	cat_1=Shuffle(1:p.tot_training_cat);
	cat_2=Shuffle(1:p.tot_training_cat);
	sessions_cat{1}=cat_1(1:p.training_cat);
	sessions_cat{2}=cat_1(p.training_cat+1:end);
	sessions_cat{3}=cat_2(1:p.training_cat);
	sessions_cat{4}=cat_2(p.training_cat+1:end);
	
	for ses=1:p.ses

		%% Establishing the stimuli images order
		vector_img_cat=repelem(sessions_cat{ses}, p.img_per_cat);
		vector_img_num=repmat(1:10, [1,p.training_cat]);
		vector_img_cat=repelem(vector_img_cat, p.rep_training);
		vector_img_num=repelem(vector_img_num, p.rep_training);
		idx=Shuffle(1:p.tot_n_training*p.rep_training);
		vector_img_cat=vector_img_cat(idx);
		vector_img_num=vector_img_num(idx);
		train_order=vector_img_cat*10-10+vector_img_num;

		% Adding the extra trials
		extra_trl=Shuffle(train_order);
		extra_trl=extra_trl(1:p.extra_train_rep);
		train_order=[train_order extra_trl];

		% Establishing, per each block, the 6 sequences in which the targets will be.
		target_sequences=zeros(p.training_blocks, p.targets_per_block);
		for x=1:size(target_sequences,1)
			tar=Shuffle(1:p.sequences_per_block);
			tar=tar(1:p.targets_per_block);
			tar=sort(tar);
			target_sequences(x,:)=tar;
		end

		% Establishing, per each target sequence, the position in which the target image will be
		for x=1:size(target_sequences,1)*size(target_sequences,2)
			target_order(x)=randi([2 p.img_per_sequence]);
		end

		% Creating the 3-D matrix with the trial images
		stimOrder_all=zeros([p.img_per_sequence,p.sequences_per_block,p.training_blocks]);

		train_counter=1;
		target_counter_order=1;

		for b=1:p.training_blocks % blocks
			target_counter_sequences=1;
			for s=1:p.sequences_per_block % sequences
				target_occurrence=0;
				for i=1:p.img_per_sequence % images

					% Target trials
					if s==target_sequences(b,target_counter_sequences) && i==target_order(target_counter_order) && target_occurrence==0
						stimOrder_all(i,s,b)=0;
						target_occurrence=1;
						if target_counter_sequences<p.targets_per_block
							target_counter_sequences=target_counter_sequences+1;
						end
						if target_counter_order<length(target_order)
							target_counter_order=target_counter_order+1;
						end
						% Stimuli trials
					else
							stimOrder_all(i,s,b)=train_order(train_counter);
							train_counter=train_counter+1;
					end
				end
			end
		end


%% Dividing the stimuli order into partitions of 3 runs/blocks each
		stimOrder=zeros(p.img_per_sequence,p.sequences_per_block,3,5);

		for par=1:5
			idx=par*3;
			stimOrder(:,:,:,par)=stimOrder_all(:,:,idx-2:idx);
		end

		
%% Saving the stimuli order of each subject
		save([saveDir, '/stimOrder_sub-', num2str(sub), '_sess-', num2str(ses)],'stimOrder')

	end
end

