function plot_FaultCurves(budgetPcts, curves_cell, apfds, labels)
    
    colors = {'#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2', '#7E6148'};
    styles = {'-', '-', '--', '-.', '-', '--', '-.', ':'};
    
    fig=figure('Name','Fault Detection Curves','Position',[100 100 680 520],'Color','w');
    
    ax_main = axes('Position', [0.12 0.14 0.83 0.80]);
    hold(ax_main, 'on');

    for i = 1:length(curves_cell)
        disp_label = sprintf('%s (CDW-APFD: %.3f)', labels{i}, apfds(i));
        plot(ax_main, budgetPcts, curves_cell{i}, styles{i}, 'Color', colors{i}, ...
             'LineWidth', 2.2, 'DisplayName', disp_label);
    end

    set(ax_main, 'FontSize', 14, 'FontName', 'Arial', 'TickDir', 'out', ...
        'LineWidth', 1.2, 'Box', 'on', 'XMinorTick', 'off', 'YMinorTick', 'off', ...
        'XTick', 0:20:100, 'YTick', 0:20:100);
    

        xlabel(ax_main, 'Test Execution Budget (%)', 'FontSize', 18);
    ylabel(ax_main, 'Diversity-Weighted Faults Detected (%)', 'FontSize', 18);
    xlim(ax_main, [0 100]); ylim(ax_main, [0 100]);
    
    leg = legend(ax_main, 'Location', 'southeast', 'FontSize', 14, 'FontName', 'Arial');
    leg.ItemTokenSize = [25, 18]; 
    leg.Box = 'off';
    
    zoom_x_max = 6;
    
    max_y_inset = 0;
    for i = 1:length(curves_cell)
        idx = find(budgetPcts <= zoom_x_max, 1, 'last');
        if ~isempty(idx)
            max_y_inset = max(max_y_inset, curves_cell{i}(idx));
        end
    end
   
    zoom_y_max = ceil((max_y_inset + 2) / 5) * 5; 
    if zoom_y_max < 5, zoom_y_max = 5; end
    
   
    ax_inset = axes('Position', [0.16 0.63 0.24 0.26]);
    hold(ax_inset, 'on');
    box(ax_inset, 'on');
   
    for i = 1:length(curves_cell)
        plot(ax_inset, budgetPcts, curves_cell{i}, styles{i}, 'Color', colors{i}, 'LineWidth', 1.8);
    end
    
   
    xlim(ax_inset, [0 zoom_x_max]);
    ylim(ax_inset, [0 zoom_y_max]);
    set(ax_inset, 'FontSize', 10, 'FontName', 'Arial', 'TickDir', 'in', ...
        'LineWidth', 1.0, 'XTick', 0:2:6);
    title(ax_inset, 'Early Discovery (0-6%)', 'FontSize', 11, 'FontName', 'Arial', 'FontWeight', 'normal');
    
    rectangle(ax_main, 'Position', [0, 0, zoom_x_max, zoom_y_max], ...
              'EdgeColor', [0.4 0.4 0.4], 'LineStyle', '--', 'LineWidth', 1.2);
          
    hold(ax_main, 'off');
    hold(ax_inset, 'off');
end