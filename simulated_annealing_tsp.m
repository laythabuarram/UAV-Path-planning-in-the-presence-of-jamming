%% Simulated Annealing TSP - Single Window Version
clc; clear; close all;

%% --- 1. Generate cities ---
N = 20;
cities = rand(N,2) * 100;

%% --- 2. Run Simulated Annealing ---
[best_route, best_cost] = simulated_annealing_tsp_single_window(cities);


%% ==========================================================
%% --- Simulated Annealing Function (SINGLE FIGURE) ---
%% ==========================================================
function [best_route, best_cost] = simulated_annealing_tsp_single_window(cities)

N = size(cities,1);

current_route = randperm(N);
current_cost  = calculate_cost(current_route, cities);

best_route = current_route;
best_cost  = current_cost;

T_initial = 1000;
T_min     = 1e-3;
alpha     = 0.995;
max_iter_per_temp = 50 * N;
update_every = 10;

%% -------- Single window layout --------
figure('Name','Simulated Annealing TSP','NumberTitle','off');
t = tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

%% ---- Left: TSP Animation ----
ax1 = nexttile(t,1);
hold(ax1,'on');

plot(ax1, cities(:,1), cities(:,2), 'bo', ...
    'MarkerSize', 8, 'MarkerFaceColor', 'b');

h_line = plot(ax1, nan, nan, 'r-', 'LineWidth', 2);

h_title = title(ax1,'');
xlabel(ax1,'X'); ylabel(ax1,'Y');
grid(ax1,'on');
axis(ax1,'equal');
xlim(ax1,[0 100]); ylim(ax1,[0 100]);

%% ---- Right: Convergence plot ----
ax2 = nexttile(t,2);
h_conv = plot(ax2, nan, nan, 'k', 'LineWidth', 2);
xlabel(ax2,'Iteration');
ylabel(ax2,'Best Cost');
title(ax2,'Convergence');
grid(ax2,'on');

%% -------- Initial draw --------
update_plot(h_line, h_title, cities, current_route, ...
            T_initial, current_cost, best_cost);
drawnow;

%% -------- Simulated Annealing --------
T = T_initial;
iter = 0;
history = [];

while T > T_min
    for i = 1:max_iter_per_temp

        % 2-opt move
        idx1 = randi(N-1);
        idx2 = randi([idx1+1, N]);

        neighbor_route = current_route;
        neighbor_route(idx1:idx2) = neighbor_route(idx2:-1:idx1);

        neighbor_cost = calculate_cost(neighbor_route, cities);
        delta_E = neighbor_cost - current_cost;

        if delta_E < 0 || log(rand) < -delta_E / T
            current_route = neighbor_route;
            current_cost  = neighbor_cost;

            if current_cost < best_cost
                best_route = current_route;
                best_cost  = current_cost;
            end
        end

        iter = iter + 1;

        if mod(iter, update_every) == 0 || i == 1
            % Update TSP plot
            update_plot(h_line, h_title, cities, current_route, ...
                        T, current_cost, best_cost);

            % Update convergence plot
            history(end+1) = best_cost; %#ok<AGROW>
            set(h_conv, 'XData', 1:numel(history), ...
                        'YData', history);

            drawnow;
        end
    end
    T = T * alpha;
end

%% -------- Final update --------
update_plot(h_line, h_title, cities, best_route, ...
            T, best_cost, best_cost);

title(ax1, sprintf('FINAL - Best Cost: %.2f', best_cost));

end


%% ==========================================================
%% --- Cost function ---
%% ==========================================================
function cost = calculate_cost(route, cities)
points = cities(route,:);
points = [points; points(1,:)];
diffs  = diff(points,1,1);
cost   = sum(sqrt(sum(diffs.^2,2)));
end


%% ==========================================================
%% --- Plot update ---
%% ==========================================================
function update_plot(h_line, h_title, cities, route, T, current_cost, best_cost)

x = cities(route,1);
y = cities(route,2);
x = [x; x(1)];
y = [y; y(1)];

set(h_line,'XData',x,'YData',y);

set(h_title,'String',{ ...
    sprintf('Temperature: %.2f', T), ...
    sprintf('Current Cost: %.2f', current_cost), ...
    sprintf('Best Cost: %.2f', best_cost) });

end
