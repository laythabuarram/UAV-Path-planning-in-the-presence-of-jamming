
clear; clc; close all;

grid_size = [20, 20];
start_pos = [1, 1];
goal_pos  = [20, 20];
n_ants = 30;
n_iterations = 150;
alpha = 1.0;
beta  = 5.0;
evaporation_rate = 0.5;
Q = 100;
max_steps = prod(grid_size);

grid = zeros(grid_size);
grid(6, 5:15)   = 1;
grid(15, 6:16)  = 1;
grid(3:12, 10)  = 1;
grid(12:18, 15) = 1;


pheromone = ones(grid_size) * 0.1;
pheromone(grid == 1) = 0;


directions = [-1 0; 1 0; 0 -1; 0 1];


best_path = [];
best_length = Inf;
history_best_lengths = nan(n_iterations,1);

figure('Position',[100 100 1200 600],'Color',[0.97 0.97 0.97]);

ax1 = subplot(1,2,1);

grid_cmap = [0.95 0.95 0.95;  
             0.4  0.4  0.4];   
imagesc(grid);
colormap(ax1, grid_cmap);
caxis([0 1]);
axis equal tight;
set(ax1,'YDir','normal','Color',[0.98 0.98 0.98]);
xlabel('Column'); ylabel('Row');
title('ACO Path Planning - Professional Visualization');


ylim(ax1, [0.5 grid_size(1)+0.5]);  
xlim(ax1, [0.5 grid_size(2)+0.5]);  

hold on;

start_plot = plot(start_pos(2),start_pos(1),'s', ...
    'Color',[0.0 0.6 0.3],'MarkerSize',12,'LineWidth',2, ...
    'MarkerFaceColor',[0.0 0.6 0.3]);

goal_plot = plot(goal_pos(2),goal_pos(1),'s', ...
    'Color',[0.8 0.2 0.2],'MarkerSize',12,'LineWidth',2, ...
    'MarkerFaceColor',[0.8 0.2 0.2]);


path_plot = plot(NaN,NaN,'-','Color',[0.0 0.3 0.7],'LineWidth',3);


ants_plots = gobjects(n_ants,1);
for k = 1:n_ants
    ants_plots(k) = plot(NaN,NaN,'o', ...
        'Color',[0.9 0.5 0.1],'MarkerSize',8,'LineWidth',1.5, ...
        'MarkerFaceColor',[0.9 0.5 0.1]);
end

pheromone_cmap = [linspace(0.95,0.2,64)', ...    
                  linspace(0.95,0.5,64)', ...    
                  linspace(1.0,0.8,64)'];        
colormap(ax1, pheromone_cmap);

pheromone_surf = surf(ax1,pheromone, ...
    'EdgeColor','none','FaceAlpha',0.4);

text(start_pos(2), start_pos(1)-0.5, 'START', ...
    'VerticalAlignment', 'top', 'HorizontalAlignment', 'center', ...
    'FontWeight', 'bold', 'Color', [0.0 0.6 0.3], 'FontSize', 9, ...
    'BackgroundColor', [1 1 1 0.7], 'EdgeColor', [0.0 0.6 0.3]);

text(goal_pos(2), goal_pos(1)+0.5, 'GOAL', ...
    'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center', ...
    'FontWeight', 'bold', 'Color', [0.8 0.2 0.2], 'FontSize', 9, ...
    'BackgroundColor', [1 1 1 0.7], 'EdgeColor', [0.8 0.2 0.2]);

legend([start_plot goal_plot path_plot ants_plots(1)], ...
    {'Start','Goal','Best Path','Ants'}, ...
    'Location','bestoutside', 'Box', 'off');

ax2 = subplot(1,2,2);
conv_plot = plot(history_best_lengths,'LineWidth',2.5, ...
    'Color',[0.5 0.2 0.6]);
xlabel('Iteration');
ylabel('Best Path Length');
title('Convergence History');
grid on;
ax2.GridAlpha = 0.2;
ax2.GridColor = [0.85 0.85 0.85];
set(ax2,'Color',[1 1 1]); % White background
xlim([1 n_iterations]);


ylim([0 grid_size(1)*2]);  % Same height scale as left plot

cb = colorbar(ax1);
cb.Label.String = 'Pheromone Intensity';
cb.Label.FontSize = 10;
cb.Color = [0.2 0.2 0.2]; % Dark text for colorbar

set(ax1, 'Position', [0.07 0.15 0.38 0.75]);  % [left bottom width height]
set(ax2, 'Position', [0.55 0.15 0.38 0.75]);  % Same height as ax1

for iter = 1:n_iterations

    all_paths = cell(n_ants,1);
    ant_positions = zeros(n_ants,2);

    for ant = 1:n_ants

        current_pos = start_pos;
        path = current_pos;
        visited = false(grid_size);
        visited(current_pos(1),current_pos(2)) = true;
        steps = 0;

        while ~isequal(current_pos,goal_pos) && steps < max_steps
            steps = steps + 1;

            neighbors = zeros(4,2);
            probs = zeros(4,1);
            count = 0;

            for d = 1:4
                next_pos = current_pos + directions(d,:);
                if next_pos(1)>=1 && next_pos(1)<=grid_size(1) && ...
                   next_pos(2)>=1 && next_pos(2)<=grid_size(2) && ...
                   grid(next_pos(1),next_pos(2))==0 && ...
                   ~visited(next_pos(1),next_pos(2))

                    count = count + 1;
                    neighbors(count,:) = next_pos;

                    tau = pheromone(next_pos(1),next_pos(2));
                    eta = 1/(norm(next_pos-goal_pos)+eps);
                    probs(count) = tau^alpha * eta^beta;
                end
            end

            if count == 0
                path = [];
                break;
            end

            probs = probs(1:count);
            neighbors = neighbors(1:count,:);
            probs = probs / sum(probs);

            next_pos = neighbors(find(cumsum(probs)>=rand(),1),:);

            path = [path; next_pos];
            current_pos = next_pos;
            visited(current_pos(1),current_pos(2)) = true;
        end

        all_paths{ant} = path;
        ant_positions(ant,:) = current_pos;

        if ~isempty(path) && isequal(path(end,:),goal_pos)
            len = size(path,1)-1;
            if len < best_length
                best_length = len;
                best_path = path;
            end
        end
    end

    pheromone = pheromone * (1-evaporation_rate);
    pheromone(grid==1) = 0;

    for ant = 1:n_ants
        path = all_paths{ant};
        if ~isempty(path) && isequal(path(end,:),goal_pos)
            len = size(path,1)-1;
            delta = Q / (len^2);
            for i = 2:size(path,1)
                r = path(i,1); c = path(i,2);
                pheromone(r,c) = pheromone(r,c) + delta;
            end
        end
    end

    history_best_lengths(iter) = best_length;

    if ~isempty(best_path)
        set(path_plot,'XData',best_path(:,2),'YData',best_path(:,1));
    end

    for k = 1:n_ants
        set(ants_plots(k),'XData',ant_positions(k,2), ...
                          'YData',ant_positions(k,1));
    end

    set(pheromone_surf,'ZData',pheromone);
    
    max_ph = max(pheromone(:));
    if max_ph > 0
        caxis(ax1,[0 max_ph*0.9]);
    end

    set(conv_plot,'YData',history_best_lengths);
    
    finite_vals = history_best_lengths(isfinite(history_best_lengths));
    if ~isempty(finite_vals)
        current_min = min(finite_vals);
        current_max = max(finite_vals);
        
        if current_max > current_min
            suggested_max = max(current_max*1.1, grid_size(1)*1.5);
            ylim(ax2,[0 suggested_max]);
        end
    end

    title(ax1,sprintf('Iteration %d/%d  |  Best Path Length: %d', ...
        iter, n_iterations, best_length));
    
    title(ax2,sprintf('Current Best: %d', best_length));
    
    drawnow limitrate;
    pause(0.05);
end

title(ax1,sprintf('Final Solution  |  Optimal Path Length: %d',best_length));
title(ax2,'Convergence Complete');

text(n_iterations*0.7, max(ylim)*0.9, sprintf('Final: %d', best_length), ...
    'Parent', ax2, 'FontSize', 10, 'FontWeight', 'bold', ...
    'Color', [0.5 0.2 0.6], 'BackgroundColor', [1 1 1 0.8]);

grid(ax2, 'on');
grid(ax2, 'minor');
ax2.MinorGridAlpha = 0.1;
ax2.MinorGridColor = [0.9 0.9 0.9];

disp('ACO optimization completed!');

disp(['Final best path length: ' num2str(best_length)]);
