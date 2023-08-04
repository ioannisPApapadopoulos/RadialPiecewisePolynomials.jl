% This script is adapted from Fortunato and Townsend's script (cylinderplot)
% for plotting solutions on a cylinder. This can be found at
% https://github.com/danfortunato/fast-poisson-solvers/blob/master/code/vis/cylinderplot.m

r  = load("r.log");
th = load("theta.log");
z  = load("z.log");

n1 = length(r);
n2 = length(th);
n3 = length(z);

[tt, rr, zz] = meshgrid(th, r, z);

% Slices in the cylinder to plot
rslice = 0;
tslice = tt(1,[1 floor(n2/4)+1 floor(n2/2)+1 floor(3*n2/4)+1],1);
zslice = squeeze(zz(1,1,[floor(n3/4)+1 floor(n3/2)+1 floor(3*n3/4)+1]));

vals = reshape(load("vals.log"), n1, n2,n3);
hslicer = slice(tt,rr,zz,vals,tslice,rslice,zslice);
hold on
for j = 1:numel(hslicer)
    h = hslicer(j);
    [xs,ys,zs] = pol2cart(h.XData,h.YData,h.ZData);
    surf(xs,ys,zs,h.CData,'EdgeColor','none','FaceColor','Interp');
end
delete(hslicer)
axis([-1 1 -1 1 -1 1])
daspect([1 1 1])
hold off

set(gca, 'Position', [0 0 1 1], 'CameraViewAngleMode', 'Manual')
colorbar('FontSize', 16, 'Position', [0.84 0.09 0.04 0.8])
axis off