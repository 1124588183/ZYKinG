% Learning-based rigid-tube RMPC trajectory tracking demo for hovercraft
% Uses hovercraft.m as the plant model for simulation.
clear; clc; close all;

%% Simulation settings
dt = 0.1;
sim_time = 120;
steps = floor(sim_time / dt);
t = (0:steps-1)' * dt;

%% Hovercraft model globals (minimal configuration)
global Vwind Betawind vd_h rd_h
Vwind = 0;
Betawind = 0;
vd_h = 0;
rd_h = 0;

%% Nominal linear model for RMPC (surge/yaw with kinematics)
% State: [x; y; psi; u; r]
% Input: [a_u; a_r] (desired accelerations)
tau_u = 6.0;  % surge time constant
tau_r = 4.0;  % yaw rate time constant
u_bias = 0;
r_bias = 0;

A = [1, 0, 0, dt, 0;
     0, 1, 0, 0,  dt;
     0, 0, 1, 0,  dt;
     0, 0, 0, 1 - dt / tau_u, 0;
     0, 0, 0, 0, 1 - dt / tau_r];

B = [0, 0;
     0, 0;
     0, 0;
     dt / tau_u, 0;
     0, dt / tau_r];

%% Tube RMPC settings
N = 20; % horizon
Q = diag([10, 10, 5, 2, 2]);
R = diag([0.5, 0.5]);
P = Q;

% Input/state constraints (nominal)
u_min = [-1.2; -0.4];
u_max = [ 1.2;  0.4];
x_min = [-inf; -inf; -inf;  0.0; -0.6];
x_max = [ inf;  inf;  inf; 20.0;  0.6];

% Disturbance bound for tube tightening (learned residual bound)
w_bound = [0.2; 0.2; 0.02; 0.4; 0.08];
tight_x = w_bound;
tight_u = [0.2; 0.1];

%% Reference S-curve
v_ref = 8;     % m/s
amp = 20;      % m
wave = 80;     % m
ref = s_curve_ref(t, v_ref, amp, wave);

%% Generate data for learning model error
data_steps = 800;
[X_train, Y_train] = generate_training_data(data_steps, dt, tau_u, tau_r);
model = train_disturbance_net(X_train, Y_train);

%% Simulation state initialization
x = zeros(5, steps);
u_cmd = zeros(2, steps);

% Full hovercraft state for plant
y_full = zeros(8, 1); % [u v p r x y phi psi]
y_full(5) = 0;
y_full(6) = 0;
y_full(8) = 0;

%% Pre-allocate logs
x_full_log = zeros(8, steps);
dist_hat = zeros(5, steps);

%% LQR ancillary controller for tube stabilization
[K_lqr, ~, ~] = dlqr(A, B, Q, R);

%% Main loop
for k = 1:steps
    x(:, k) = [y_full(5); y_full(6); y_full(8); y_full(1); y_full(4)];
    x_ref = ref(:, k);

    % Disturbance prediction from NN (nominal residual)
    dist_hat(:, k) = predict_disturbance(model, x(:, k), u_cmd(:, max(k-1,1)));

    % Solve nominal MPC for tracking (tube tightened)
    u_nom = solve_mpc(A, B, Q, R, P, N, x(:, k) - dist_hat(:, k), x_ref, ...
        x_min + tight_x, x_max - tight_x, u_min + tight_u, u_max - tight_u);

    % Tube feedback law
    u = u_nom + (-K_lqr * (x(:, k) - x_ref));
    u = min(max(u, u_min), u_max);
    u_cmd(:, k) = u;

    % Map to hovercraft inputs (12-element)
    ui = zeros(12, 1);
    % Map acceleration commands into propeller pitch (B1/B2)
    base_thrust = 8 + 8 * u(1);
    diff_thrust = 3 * u(2);
    ui(1) = base_thrust + diff_thrust;
    ui(2) = base_thrust - diff_thrust;
    ui(3) = 0; % rudder left
    ui(4) = 0; % rudder right
    ui(5:12) = 0;

    % Simulate plant using hovercraft model
    ydot = hovercraft(y_full, ui);
    y_full = y_full + dt * ydot;
    x_full_log(:, k) = y_full;
end

%% Visualization
figure('Name', 'S-curve tracking');
plot(ref(1, :), ref(2, :), 'k--', 'LineWidth', 1.5); hold on;
plot(x(1, :), x(2, :), 'b', 'LineWidth', 1.5);
axis equal; grid on;
xlabel('x (m)'); ylabel('y (m)');
legend('Reference', 'Hovercraft');
title('Trajectory tracking');

figure('Name', 'States and inputs');
subplot(3, 1, 1);
plot(t, x(4, :), 'b', t, ref(4, :), 'k--', 'LineWidth', 1.2);
grid on; ylabel('u (m/s)');
legend('Actual', 'Ref');
title('Speed');

subplot(3, 1, 2);
plot(t, x(3, :), 'b', t, ref(3, :), 'k--', 'LineWidth', 1.2);
grid on; ylabel('\psi (rad)');
legend('Actual', 'Ref');
title('Heading');

subplot(3, 1, 3);
plot(t, u_cmd(1, :), 'b', t, u_cmd(2, :), 'r', 'LineWidth', 1.2);
grid on; ylabel('u_{cmd}');
xlabel('Time (s)');
legend('a_u', 'a_r');
title('Control inputs');

figure('Name', 'Disturbance estimate');
plot(t, dist_hat(4, :), 'LineWidth', 1.2);
grid on;
xlabel('Time (s)'); ylabel('d_u');
title('Learned model error estimate');

%% Local functions
function ref = s_curve_ref(t, v_ref, amp, wave)
    x = v_ref * t;
    y = amp * sin(2 * pi * x / wave);
    dy_dx = (2 * pi * amp / wave) * cos(2 * pi * x / wave);
    psi = atan2(dy_dx, 1);
    u = v_ref * ones(size(t));
    r = gradient(psi, t(2) - t(1));
    ref = [x'; y'; psi'; u'; r'];
end

function [X_train, Y_train] = generate_training_data(steps, dt, tau_u, tau_r)
    X_train = zeros(7, steps);
    Y_train = zeros(5, steps);
    y_full = zeros(8, 1);
    for k = 1:steps
        x = [y_full(5); y_full(6); y_full(8); y_full(1); y_full(4)];
        u = [0.6 * sin(0.01 * k); 0.3 * cos(0.008 * k)];
        ui = zeros(12, 1);
        base_thrust = 8 + 8 * u(1);
        diff_thrust = 3 * u(2);
        ui(1) = base_thrust + diff_thrust;
        ui(2) = base_thrust - diff_thrust;
        ui(3:12) = 0;

        ydot = hovercraft(y_full, ui);
        y_full_next = y_full + dt * ydot;
        x_next = [y_full_next(5); y_full_next(6); y_full_next(8); y_full_next(1); y_full_next(4)];

        A_nom = [1, 0, 0, dt, 0;
                 0, 1, 0, 0,  dt;
                 0, 0, 1, 0,  dt;
                 0, 0, 0, 1 - dt / tau_u, 0;
                 0, 0, 0, 0, 1 - dt / tau_r];
        B_nom = [0, 0;
                 0, 0;
                 0, 0;
                 dt / tau_u, 0;
                 0, dt / tau_r];

        x_nom_next = A_nom * x + B_nom * u;
        dist = x_next - x_nom_next;

        X_train(:, k) = [x; u];
        Y_train(:, k) = dist;
        y_full = y_full_next;
    end
end

function model = train_disturbance_net(X_train, Y_train)
    if exist('fitnet', 'file') == 2
        net = fitnet(10, 'trainlm');
        net.trainParam.showWindow = false;
        net = train(net, X_train, Y_train);
        model.type = 'fitnet';
        model.net = net;
    else
        % fallback to linear regression
        W = Y_train / X_train;
        model.type = 'linear';
        model.W = W;
    end
end

function d = predict_disturbance(model, x, u)
    xu = [x; u];
    if strcmp(model.type, 'fitnet')
        d = model.net(xu);
    else
        d = model.W * xu;
    end
end

function u = solve_mpc(A, B, Q, R, P, N, x0, x_ref, x_min, x_max, u_min, u_max)
    nx = size(A, 1);
    nu = size(B, 2);

    H = zeros(N * nu);
    f = zeros(N * nu, 1);
    Aineq = [];
    bineq = [];

    x = x0;
    for k = 1:N
        H((k-1)*nu+1:k*nu, (k-1)*nu+1:k*nu) = R;
    end
    H = H + H';

    % Build prediction matrices
    Phi = zeros(nx * N, nx);
    Gamma = zeros(nx * N, nu * N);
    for i = 1:N
        Phi((i-1)*nx+1:i*nx, :) = A^i;
        for j = 1:i
            Gamma((i-1)*nx+1:i*nx, (j-1)*nu+1:j*nu) = A^(i-j) * B;
        end
    end

    Qbar = kron(eye(N), Q);
    Qbar(end-nx+1:end, end-nx+1:end) = P;
    H = Gamma' * Qbar * Gamma + kron(eye(N), R);
    f = Gamma' * Qbar * (Phi * x0 - repmat(x_ref, N, 1));

    % Input constraints
    Aineq = [eye(nu*N); -eye(nu*N)];
    bineq = [repmat(u_max, N, 1); -repmat(u_min, N, 1)];

    % State constraints
    Ax = [Gamma; -Gamma];
    bx = [repmat(x_max, N, 1) - Phi * x0;
          -(repmat(x_min, N, 1) - Phi * x0)];
    Aineq = [Aineq; Ax];
    bineq = [bineq; bx];

    opts = optimoptions('quadprog', 'Display', 'off');
    u_seq = quadprog(H, f, Aineq, bineq, [], [], [], [], [], opts);
    if isempty(u_seq)
        u = zeros(nu, 1);
    else
        u = u_seq(1:nu);
    end
end
