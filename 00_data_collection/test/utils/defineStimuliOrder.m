%% This function establishes the randomized stimuli order
function stimOrder=defineStimuliOrder(p)

% Establishing the stimuli images order
test_order=imgOrder(p);

% Establishing, per each block, the 6 sequences in which the targets will be.
target_sequences=zeros(p.test_blocks, p.targets_per_block);
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
stimOrder=zeros([p.img_per_sequence,p.sequences_per_block,p.test_blocks]);

test_counter=1;
target_counter_order=1;

for b=1:p.test_blocks % blocks
    target_counter_sequences=1;
    for s=1:p.sequences_per_block % sequences
        target_occurrence=0;
        for i=1:p.img_per_sequence % images

            % Target trials
            if s==target_sequences(b,target_counter_sequences) && i==target_order(target_counter_order) && target_occurrence==0
                stimOrder(i,s,b)=0;
                target_occurrence=1;
                if target_counter_sequences<p.targets_per_block
                    target_counter_sequences=target_counter_sequences+1;
                end
                if target_counter_order<length(target_order)
                    target_counter_order=target_counter_order+1;
                end
                % Stimuli trials
            else
                stimOrder(i,s,b)=test_order(test_counter);
                test_counter=test_counter+1;
            end
        end
    end
end
end

