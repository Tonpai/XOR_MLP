clc;
clear;
close all;

% XOR Problem
x = [
    1 1;
    1 0;
    0 1;
    0 0;
];
t = [0 1 1 0]';

n = 0.5;

feature_number = size(x,2);

w_hidden_node_number = 2;
w_output_node_number= 1; % regression problem

w_hidden = rand(feature_number, w_hidden_node_number);
w_hidden_bias = rand(1, w_hidden_node_number);

w_output = rand(w_hidden_node_number, w_output_node_number);
w_output_bias = rand(1, w_output_node_number);

for k = 1:1000
    for i = 1:size(t, 1)
        y_hidden = tanh(x(i,:) * w_hidden + w_hidden_bias);
        y_output = logsig(y_hidden * w_output + w_output_bias);
        
        e = t(i) - y_output;
        
        d_output = e.* y_output .* (1 - y_output);
        d_hidden = (1-y_hidden.^2) .* (d_output * w_output');
        
        w_output = w_output + n * y_hidden' * d_output;
        w_output_bias = w_output_bias + n * d_output;
        w_hidden = w_hidden + n * x (i, :)' * d_hidden;
        w_hidden_bias = w_hidden_bias +  n * d_hidden;
    end
    
    [X1, X2] = meshgrid(-0.5:1.5);
    Y1 = w_hidden_bias(1) + X1*w_hidden(1, 1) + X2 * w_hidden(2, 1);
    Y2 = w_hidden_bias(2) + X1*w_hidden(1, 2) + X2 * w_hidden(2, 2);
    plot(x(1:end,1), x(1:end,2), 'ro');
    hold on;
    contour(X1, X2, Y1, [0,0], 'k');
    contour(X1, X2, Y2, [0,0], 'k');
    title(['Iteration: ' num2str(k)]);
    hold off
    drawnow;
end

y_hidden = logsig(x * w_hidden + w_hidden_bias);
y_output = logsig(y_hidden * w_output + w_output_bias);