function pc2jpg(pc, name)
    s = max(abs(pc(:)));
    r = pc(1,:)';
    r = (r - min(r(:))) / max(r(:));
    g = pc(2,:)';
    g = (g - min(g(:))) / max(g(:));
    b = pc(3,:)';
    b = (b - min(b(:))) / max(b(:));
    color = 0.5 * [r, g, b];
    scatter3(pc(1,:,:), pc(2,:,:), pc(3,:,:), 10, color, 'filled');
    axis([-s/3 s/3 -s/3 s/3 -s/3 s/3]);
    axis off;
    print(name, '-djpeg', '-noui');
end
