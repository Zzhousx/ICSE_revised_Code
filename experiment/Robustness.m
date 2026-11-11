clear;
clc;
alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5]; % p_values
beta_values = [3, 4, 5, 6, 7];  % q_values

acc_data_orl = [
    0.7736, 0.7721, 0.7734, 0.7719, 0.7710;
    0.7750, 0.7716, 0.7768, 0.7735, 0.7717;
    0.7705, 0.7719, 0.7749, 0.7752, 0.7735;
    0.7711, 0.7739, 0.7762, 0.7757, 0.7749;
    0.7668, 0.7715, 0.7777, 0.7759, 0.7761
]*100;



[X, Y] = meshgrid(1:length(beta_values), 1:length(alpha_values));

figure('Position', [100, 100, 700, 550]); 


s = surf(X, Y, acc_data_orl);

shading interp; 
colormap('jet'); 
colorbar;
view(-45, 20); 

set(gca, 'XTick', 1:length(beta_values), 'XTickLabel', beta_values);
set(gca, 'YTick', 1:length(alpha_values), 'YTickLabel', alpha_values);
set(gca, 'FontSize', 15);

xlim([1, length(beta_values)]);
ylim([1, length(alpha_values)]);
zlim([72, 80]);
ax = gca;

font_name = 'Arial';
font_size_axis = 16;
font_size_label = 20;
xlabel('\tau', 'FontName', font_name, 'FontSize', font_size_label, 'FontWeight', 'bold');
ylabel('\alpha', 'FontName', font_name, 'FontSize', font_size_label, 'FontWeight', 'bold');
zlabel('CDW-APFD (%)', 'FontSize', 22);
box on;
grid on;
