%% PSO with Animation + GIF Export + Partial Clipping (No Obstacles)
clear; clc; close all;
%% ------------------- Parameters -------------------
goal        = [9, 9];          % Target position
l_bound     = [-10, -10];
u_bound     = [10, 10];
swarmsize   = 40;
dimension   = 2;
iterations  = 70;
omega       = 0.7;
phip        = 1.5;
phig        = 1.5;
%% ------------------- Objective Function -------------------
objective = @(x) objective_without_obstacle(x, goal);
%% ------------------- Run PSO -------------------
[pos_arr, pos_local, pos_vt, best_scores, final_gbest_pos] = PSO( ...
    objective, l_bound, u_bound, swarmsize, dimension, iterations, ...
    omega, phip, phig, goal);
fprintf('\nFinal Best Position: [%.4f , %.4f]\n', ...
        final_gbest_pos(1), final_gbest_pos(2));
%% =================== PSO FUNCTION ===================
function [pos_arr, pos_local, pos_vt, best_scores, final_gbest_pos] = PSO( ...
    objective, l_bound, u_bound, swarmsize, dimension, iterations, ...
    omega, phip, phig, goal)
    ns = swarmsize;
    nd = dimension;
    %% -------- Initialize particles (random) --------
    arr = zeros(ns, nd);
    for i = 1:ns
        for d = 1:nd
            arr(i,d) = unifrnd(l_bound(d), u_bound(d));
        end
    end
    Vt = zeros(ns, nd);
    best_local = arr;
    obj_local = zeros(ns,1);
    for i = 1:ns
        obj_local(i) = objective(arr(i,:));
    end
    [obj_global, idx] = min(obj_local);
    best_global = best_local(idx,:);
    pos_arr = cell(1, iterations+1);
    pos_local = cell(1, iterations+1);
    pos_vt = cell(1, iterations);
    best_scores = zeros(1, iterations);
    pos_arr{1} = arr;
    pos_local{1} = best_local;
    %% ------------------- Visualization -------------------
    figure('Name','PSO Animation','NumberTitle','off');
    axis([l_bound(1) u_bound(1) l_bound(2) u_bound(2)]);
    axis equal;
    grid on; hold on;
    xlabel('X'); ylabel('Y');
    set(gca,'Layer','top');
    % Contour (distance to goal)
    [X,Y] = meshgrid(linspace(l_bound(1),u_bound(1),120));
    Z = (X-goal(1)).^2 + (Y-goal(2)).^2;
    contourf(X,Y,Z,30,'LineColor','none','FaceAlpha',0.85);
    colormap(parula); colorbar;
    % Goal
    plot(goal(1), goal(2), 'rp', ...
        'MarkerSize',18,'MarkerFaceColor','r');
    % Particles + velocity
    h_particles = plot(arr(:,1), arr(:,2), 'bo', ...
        'MarkerFaceColor','b','MarkerSize',7);
    h_vel = quiver(arr(:,1), arr(:,2), ...
        zeros(ns,1), zeros(ns,1), ...
        0, 'r', 'LineWidth',1.5, 'MaxHeadSize',2);

    % Display initial title (Iteration 0)
    title(sprintf('Iteration 0 / %d | Best = %.6f', ...
              iterations, obj_global));
    drawnow; % Ensure the initial state is plotted before the loop starts
    
    %% ------------------- GIF settings -------------------
    gif_filename = 'PSO_selected_iterations.gif';
    gif_iters = [1 10 50 70];
    %% ------------------- Main Loop -------------------
    scale = 0.4; % طول الأسهم الأصلي
    for k = 1:iterations
        rp = rand(ns,nd);
        rg = rand(ns,nd);
        % تحديث السرعة والموقع
        Vt = omega*Vt ...
           + phip*rp.*(best_local - arr) ...
           + phig*rg.*(repmat(best_global,ns,1) - arr);
        arr = arr + Vt;
        % Apply bounds
        for d = 1:nd
            arr(:,d) = max(min(arr(:,d),u_bound(d)),l_bound(d));
        end
        % Evaluation
        obj = zeros(ns,1);
        for i = 1:ns
            obj(i) = objective(arr(i,:));
        end
        % Update pbest
        improved = obj < obj_local;
        best_local(improved,:) = arr(improved,:);
        obj_local = min(obj_local, obj);
        % Update gbest
        [new_best, idx] = min(obj_local);
        if new_best < obj_global
            obj_global = new_best;
            best_global = best_local(idx,:);
        end
        % Save history
        pos_arr{k+1} = arr;
        pos_local{k+1} = best_local;
        pos_vt{k} = Vt;
        best_scores(k) = obj_global;
        % ---------------- Update Animation ----------------
        set(h_particles,'XData',arr(:,1),'YData',arr(:,2));
        % Partial clipping arrows
        Ux = Vt(:,1)*scale;
        Uy = Vt(:,2)*scale;
        for i = 1:ns
            x_end = arr(i,1) + Ux(i);
            y_end = arr(i,2) + Uy(i);
            % X bounds
            if x_end > u_bound(1)
                alpha = (u_bound(1)-arr(i,1))/Ux(i);
                Ux(i) = Ux(i)*max(0,alpha);
                Uy(i) = Uy(i)*max(0,alpha);
            elseif x_end < l_bound(1)
                alpha = (l_bound(1)-arr(i,1))/Ux(i);
                Ux(i) = Ux(i)*max(0,alpha);
                Uy(i) = Uy(i)*max(0,alpha);
            end
            % Y bounds
            if y_end > u_bound(2)
                alpha = (u_bound(2)-arr(i,2))/Uy(i);
                Ux(i) = Ux(i)*max(0,alpha);
                Uy(i) = Uy(i)*max(0,alpha);
            elseif y_end < l_bound(2)
                alpha = (l_bound(2)-arr(i,2))/Uy(i);
                Ux(i) = Ux(i)*max(0,alpha);
                Uy(i) = Uy(i)*max(0,alpha);
            end
        end
        set(h_vel,'XData',arr(:,1),'YData',arr(:,2), ...
                  'UData',Ux,'VData',Uy);
        title(sprintf('Iteration %d / %d | Best = %.6f', ...
              k, iterations, obj_global));
        drawnow;
        pause(0.03);
        
        % -------- Save current iteration as image and to GIF --------
        
        % NEW LOGIC: Save Iteration 1 as a standalone PNG image
        if k == 1
            frame = getframe(gcf);
            imwrite(frame.cdata, 'PSO_iteration_01.png');
            fprintf('\nSwarm state for Iteration 1 saved to: PSO_iteration_01.png\n');
        end

        % Original GIF saving logic (using gif_iters, which includes k=1)
        if ismember(k, gif_iters)
            frame = getframe(gcf);
            img = frame2im(frame);
            [imind, cm] = rgb2ind(img, 256);
            if k == gif_iters(1)
                imwrite(imind, cm, gif_filename, 'gif', ...
                        'LoopCount', inf, 'DelayTime', 1);
            else
                imwrite(imind, cm, gif_filename, 'gif', ...
                        'WriteMode', 'append', 'DelayTime', 1);
            end
        end
    end
    final_gbest_pos = best_global;
end
%% =================== Objective Function ===================
function cost = objective_without_obstacle(x, goal)
    cost = norm(x - goal)^2;  % distance to goal
end