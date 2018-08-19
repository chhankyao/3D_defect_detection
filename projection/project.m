addpath('matpcl');
addpath(genpath('phone_models'));

files = dir('phone_models/*/models/*.pcd');

for i = 1 : length(files)
    f = strcat(files(i).folder, "/", files(i).name);
    data = loadpcd(f);
    
    pc = data(1:3, :);       
    pc = pc_rotate(pc, pi/3, 'x');
    pc = pc_rotate(pc, pi/3, 'y');
    pc = pc_rotate(pc, pi/3, 'z');
    
    name = strcat('projections/', int2str(i));
    pc2jpg(pc, name); 
    
%     for j = 1 : 5
%         name = strcat('projections/', int2str(i), '_', int2str(j));
%         pc2jpg(pc, name);
%         pc = pc_rotate(pc, 2*pi/180, 'x');
%         pc = pc_rotate(pc, 2*pi/180, 'y');
%         pc = pc_rotate(pc, 2*pi/180, 'z');
%     end
    sprintf('%d', i);
end
