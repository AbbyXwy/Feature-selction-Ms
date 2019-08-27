function yitiaolong_14task
%% main function 
load num_contrast_1112
label = num_con;    %
% increase subject number with interval of 100 subjects
for i = 5:5
    if i==10
        [fortrain_test,label_for] = get_volume_14task(label);
        label_f = 987;
    else
        [fortrain_test,label_for] = get_volume_14task(label(1:100*i));
        label_f = 100*i;
    end

train_data = [];
train_label = [];
sub = i*100;
    %% train and test data
    for k = 1:1
         fortrain = fortrain_test;
         forlabel = label_for;
        for l = 1:10
        train_data = [train_data;cell2mat(fortrain(l))];
        train_label = [train_label,cell2mat(forlabel(l))];
        end
        train_contrast_14task_181server(train_data,train_label,sub);
    end
end
        
            
        
        





