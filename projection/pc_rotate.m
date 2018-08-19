function pc2 = pc_rotate(pc, theta, axis)

    if strcmp(axis, 'x')
        m = [1,         0,          0;
             0,         cos(theta), -sin(theta);
             0,         sin(theta), cos(theta)];
    elseif strcmp(axis, 'y')
        m = [cos(theta), 0,         sin(theta);
             0,          1,         0;
             -sin(theta),0,         cos(theta)];
    elseif strcmp(axis, 'z')
        m = [cos(theta), -sin(theta), 0;
             sin(theta), cos(theta),  0;
             0,          0,           1];
    end
    
    pc2 = m * pc(1:3, :);
    % pc2 = [m * pc(1:3, :); m * pc(4:6, :)];
    
end
