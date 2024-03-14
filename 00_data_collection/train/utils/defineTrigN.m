function trig_1=defineTrigN(img_category,within_category_number)

	stim_N=img_category*10-10+within_category_number;
	str_N=string(stim_N);
	size_stim=strlength(str_N);
	char_N=char(str_N);

	if size_stim==1 % if 1 digit
		trig_1=stim_N;
	elseif size_stim>=2 % if 2 or more digits
		trig_1=str2double(char_N(1:2));

	end
end

