function img_order=imgOrder(p)

% Image order
img_order=1:p.tot_n_test;
img_order=repmat(img_order,[1,p.rep_test]);
img_order=Shuffle(img_order);

% Adding the extra trials
extra_trl=Shuffle(1:p.tot_n_test);
extra_trl=extra_trl(1:p.extra_test_rep);
img_order=[img_order extra_trl];
end