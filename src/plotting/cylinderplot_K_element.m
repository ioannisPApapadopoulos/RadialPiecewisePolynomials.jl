% This script is adapted from Fortunato and Townsend's script (cylinderplot)
% for plotting solutions on a cylinder. This can be found at
% https://github.com/danfortunato/fast-poisson-solvers/blob/master/code/vis/cylinderplot.m

z  = load("z.log");
n3 = length(z);

hold on

K = 2;
for k = 1:K
    r  = load("r"+k+".log");
    th = load("theta"+k+".log");

    n1 = length(r);
    n2 = length(th);
    
    vals = reshape(load("vals"+k+".log"), n1, n2,n3);
    
    [tt, rr, zz] = meshgrid(th, r, z);
    % Slices in the cylinder to plot
    rslice = 0;
    tslice = tt(1,[1 floor(n2/4)+1 floor(n2/2)+1 floor(3*n2/4)+1],1);
    zslice = squeeze(zz(1,1,[floor(n3/4)+1 floor(n3/2)+1 floor(3*n3/4)+1]));
    
    hslicer = slice(tt,rr,zz,vals,tslice,rslice,zslice);
    for j = 1:numel(hslicer)
        h = hslicer(j);
        [xs,ys,zs] = pol2cart(h.XData,h.YData,h.ZData);
        surf(xs,ys,zs,h.CData,'EdgeColor','none','FaceColor','Interp');
    end
    delete(hslicer)
end



axis([-1.1 1.1 -1.1 1.1 -1.1 1.1])
daspect([1 1 1])
%view(caz, cel);
set(gca, 'Position', [0.2 0.25 0.6 0.6], 'CameraViewAngleMode', 'Manual', 'FontSize', 25)
colorbar('FontSize', 16, 'Position', [0.84 0.09 0.04 0.8])
axis on
xlabel("$x$", "FontSize", 40, 'Interpreter','latex')
ylabel("$y$", "FontSize", 40, 'Interpreter','latex')
zlabel("$z$", "FontSize", 40, 'Interpreter','latex')
cmap = getPyPlot_cMap('bwr');
colormap(cmap)