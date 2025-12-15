clear; clc; close all;

% ==================== Parameters ====================
start_pos = [0, 0];
goal_pos  = [10, 10];
obstacles = [5, 5, 1.7];           % single obstacle in the middle

pop_size        = 80;
num_points      = 10;
max_generations = 250;
elite_count     = 8;
mutation_rate   = 0.3;

x_min = -1;  x_max = 11;
y_min = -1;  y_max = 11;

% ==================== Initialize Population ====================
population = zeros(pop_size, num_points*2);
for i = 1:pop_size
    x = linspace(start_pos(1), goal_pos(1), num_points+2);
    y = linspace(start_pos(2), goal_pos(2), num_points+2);
    x(2:end-1) = x(2:end-1) + randn(1,num_points)*4;
    y(2:end-1) = y(2:end-1) + randn(1,num_points)*4;
    x = max(x_min, min(x_max, x));
    y = max(y_min, min(y_max, y));
    population(i,:) = [x(2:end-1) y(2:end-1)];
end

best_path = [];
best_fitness_global = inf;

% ==================== Main GA Loop ====================
for gen = 1:max_generations
    
    fitness = zeros(pop_size,1);
    paths   = cell(pop_size,1);
    
    for i = 1:pop_size
        mid   = population(i,:);
        x_mid = mid(1:num_points);
        y_mid = mid(num_points+1:end);
        path_x = [start_pos(1), x_mid, goal_pos(1)];
        path_y = [start_pos(2), y_mid, goal_pos(2)];
        paths{i} = [path_x', path_y'];
        
        L = sum(sqrt(diff(path_x).^2 + diff(path_y).^2));
        
        C = 0;
        for s = 1:length(path_x)-1
            if lineIntersectsCircle([path_x(s), path_y(s)], [path_x(s+1), path_y(s+1)], obstacles)
                C = C + 1000;
            end
        end
        fitness(i) = L + C;
    end
    
    [current_best, best_idx] = min(fitness);
    if current_best < best_fitness_global
        best_fitness_global = current_best;
        best_path = paths{best_idx};
    end
    
    % ==================== Plot every 12 generations ====================
    if mod(gen,12)==0 || gen==1 || gen==max_generations || best_fitness_global < 15
        figure(1); clf; hold on; axis equal; grid on; box on;
        title(sprintf('Generation %d  →  Best Fitness = %.3f', gen, best_fitness_global), ...
              'FontSize',16);
        
        % Draw obstacle
        viscircles(obstacles(1:2), obstacles(3), 'Color','red', 'LineWidth',3);
        
        % All 80 paths of current generation — THICK GRAY LINES (very easy to see)
        for i = 1:pop_size
            plot(paths{i}(:,1), paths{i}(:,2), 'Color',[0.7 0.7 0.7], 'LineWidth', 1.5);
        end
        
        % Best path so far — thick blue
        plot(best_path(:,1), best_path(:,2), 'b-', 'LineWidth', 5);
        
        % Start (bright green circle) and Goal (magenta pentagon)
        plot(start_pos(1), start_pos(2), 'o', 'MarkerSize',18, ...
             'MarkerFaceColor',[0 1 0], 'MarkerEdgeColor','k', 'LineWidth',2);
        plot(goal_pos(1), goal_pos(2), 'p', 'MarkerSize',18, ...
             'MarkerFaceColor','magenta', 'MarkerEdgeColor','k', 'LineWidth',2);
        
        xlim([x_min x_max]); ylim([y_min y_max]);
        xlabel('X'); ylabel('Y');
        drawnow;
    end
    
    if best_fitness_global < 15
        fprintf('Excellent path found at generation %d! Fitness = %.3f\n', gen, best_fitness_global);
        break;
    end
    
    % ==================== Create Next Generation ====================
    new_population = zeros(pop_size, num_points*2);
    [~, idx] = sort(fitness);
    new_population(1:elite_count,:) = population(idx(1:elite_count),:);
    
    for i = elite_count+1:pop_size
        tour = randi(pop_size,1,6);
        [~,p1] = min(fitness(tour)); parent1 = population(tour(p1),:);
        tour = randi(pop_size,1,6);
        [~,p2] = min(fitness(tour)); parent2 = population(tour(p2),:);
        
        cp = randi(num_points*2-1);
        child = [parent1(1:cp) parent2(cp+1:end)];
        
        if rand < mutation_rate
            mut_idx = randi(num_points*2);
            if mut_idx <= num_points
                child(mut_idx) = x_min + rand*(x_max-x_min);
            else
                child(mut_idx) = y_min + rand*(y_max-y_min);
            end
        end
        new_population(i,:) = child;
    end
    population = new_population;
end

% ==================== Final Beautiful Plot ====================
figure(2); hold on; axis equal; grid on; box on;
title('Final Optimal Path','FontSize',18,'FontWeight','bold');
viscircles(obstacles(1:2), obstacles(3), 'Color','red','LineWidth',4);
plot(best_path(:,1), best_path(:,2), 'b-', 'LineWidth',7);
plot(start_pos(1),start_pos(2),'o','MarkerSize',22,'MarkerFaceColor',[0 1 0],'MarkerEdgeColor','k','LineWidth',3);
plot(goal_pos(1),goal_pos(2),'p','MarkerSize',22,'MarkerFaceColor','magenta','MarkerEdgeColor','k','LineWidth',3);
legend('Optimal Path','Start','Goal','Location','bestoutside');
xlabel('X','FontSize',14); ylabel('Y','FontSize',14);
fprintf('=== DONE === Final best path length ≈ %.3f\n', best_fitness_global);

% ==================== Collision Detection Function ====================
function intersects = lineIntersectsCircle(p1, p2, obs)
    intersects = false;
    for k = 1:size(obs,1)
        c = obs(k,1:2); r = obs(k,3);
        v = p2 - p1;
        len = norm(v);
        if len == 0, continue; end
        d = norm(cross([v 0], [p1-c 0])) / len;
        if d <= r
            t = dot(c - p1, v) / (len^2);
            if t >= 0 && t <= 1
                intersects = true;
                return;
            end
        end
    end
end