function [bars, nbar, nsub] = brickplot(data,colors,y_lim, x_lim, fontsize,mytitle, ... 
    x_label,y_label, x_values, xticklabels)

% transforms the Data matrix into cell format if needed
if iscell(data)==0
    data = num2cell(data,2);
end


% number of factors/groups/conditions
nbar = size(data,1);
% bar size
Wbar = 0.75;

% confidence interval
ConfInter = 0.95;

% color of the box + error bar
trace = [0.5 0.5 0.5];

for n = 1:nbar
    
    clear DataMatrix
    clear jitter jitterstrength
    DataMatrix = data{n,:}';
    
    % number of subjects
    nsub = length(DataMatrix(~isnan(DataMatrix)));
    
    curve = nanmean(DataMatrix);
    sem   = nanstd(DataMatrix')'/sqrt(nsub);
    mystd = nanstd(DataMatrix);
    conf  = tinv(1 - 0.5*(1-ConfInter),nsub);
    
    width = x_lim(2)/8/5;
    
    
    fill([x_values(n)-width x_values(n)+width x_values(n)+width x_values(n)-width],...
        [curve-sem curve-sem curve+sem curve+sem],...
        colors(n, :),...
        'EdgeColor', 'none',...%trace,...
        'FaceAlpha',0.5);
    hold on
    
    
    fill([x_values(n)-width x_values(n)+width x_values(n)+width x_values(n)-width],...
        [curve-sem*conf curve-sem*conf curve+sem*conf curve+sem*conf],...
        colors(n,:),...
        'EdgeColor', 'none',...%trace,...
        'FaceAlpha',0.23);
    hold on
    
    xMean = [x_values(n)-width ; x_values(n) + width];
    yMean = [curve; curve];
    plot(xMean,yMean,'-','LineWidth',2,'Color',colors(n, :));
    hold on
    

end

% axes and stuff
ylim(y_lim);

set(gca,'FontSize',fontsize,...
    'XLim', x_lim ,...
    'XTick',x_values,...
    'XTickLabel',xticklabels);

title(mytitle);
xlabel(x_label);
ylabel(y_label);












