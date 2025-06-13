clc;
clear;
%% Load data
load data.mat
data_EIS=SOC_55_80cyc;
N_mea=length(data_EIS(:,1));
f_mea=data_EIS(:,1);
w_mea=2*pi*f_mea;
Z_real=data_EIS(:,2);
Z_imag=data_EIS(:,3);
%% DRT preset value
RQ_delta_1=0.2;  
RQ_delta_2=0.1;  
RQ_delta_3=0.05; 
RQ_max=5;
RQ_min=-5;  
r1=0.5;     
r2=0.1;     
r3=0.01;    
%% Calculation of the initial parameters of DRT
N_rq_1=(RQ_max-RQ_min)/RQ_delta_1;
N_rq_2=(RQ_max-RQ_min)/RQ_delta_2;
N_rq_3=(RQ_max-RQ_min)/RQ_delta_3;
f_rq_1=10.^[RQ_min:RQ_delta_1:RQ_max-RQ_delta_1];
f_rq_2=10.^[RQ_min:RQ_delta_2:RQ_max-RQ_delta_2];
f_rq_3=10.^[RQ_min:RQ_delta_3:RQ_max-RQ_delta_3];
w_rq_1=2*pi*f_rq_1;
w_rq_2=2*pi*f_rq_2;
w_rq_3=2*pi*f_rq_3;
x_rq_1=(log10(w_rq_1))';
x_rq_2=(log10(w_rq_2))';
x_rq_3=(log10(w_rq_3))';
N_rl=1;
Init_1=0.002*ones(N_rq_1+N_rl+1,1);
Init_2=0.002*ones(N_rq_2+N_rl+1,1);
Init_3=0.002*ones(N_rq_3+N_rl+1,1);
%% DRT constructs the A matrix
A_1 = createAM(N_rq_1, w_rq_1, w_mea, N_mea);
A_2 = createAM(N_rq_2, w_rq_2, w_mea, N_mea);
A_3 = createAM(N_rq_3, w_rq_3, w_mea, N_mea);
%% The solution Settings for the least squares solution of DRT
[Aineq_1, bineq_1]=createAb(N_rq_1);
[Aineq_2, bineq_2]=createAb(N_rq_2);
[Aineq_3, bineq_3]=createAb(N_rq_3);
Aeq=[];
beq=[];
options = optimoptions('fmincon','Algorithm','sqp');
nonlcon = [];
lb = [];
ub = [];
%% DRT regularization
D1 = createM((N_rq_1)/10);
D2 = createM((N_rq_2)/10);
D3 = createM((N_rq_3)/10);
%% DRT solution  
b_cell=zeros(2*N_mea,1);
for m=1:1:N_mea
    b_cell(m,1)=Z_real(m);
end
for m=N_mea+1:1:2*N_mea
    b_cell(m,1)=Z_imag(m-N_mea);
end

fun_cell_11 = @(X) ((A_1*X - b_cell)' * (A_1*X - b_cell) + r1 * X' * D1 * X);
x_cell_1_1 = fmincon(fun_cell_11, Init_1, Aineq_1, bineq_1, Aeq, beq, lb, ub, nonlcon, options);
x_cell_11=x_cell_1_1;
x_cell_1_1(x_cell_1_1 < 0.001) = 0;

fun_cell_21 = @(X) ((A_1*X - b_cell)' * (A_1*X - b_cell) + r2 * X' * D1 * X);
x_cell_2_1 = fmincon(fun_cell_21, Init_1, Aineq_1, bineq_1, Aeq, beq, lb, ub, nonlcon, options);
x_cell_21=x_cell_2_1;
x_cell_2_1(x_cell_2_1 < 0.001) = 0;

fun_cell_22 = @(X) ((A_2*X - b_cell)' * (A_2*X - b_cell) + r2 * X' * D2 * X);
x_cell_2_2 = fmincon(fun_cell_22, Init_2, Aineq_2, bineq_2, Aeq, beq, lb, ub, nonlcon, options);
x_cell_22=x_cell_2_2;
x_cell_2_2(x_cell_2_2 < 0.001) = 0;

fun_cell_23 = @(X) ((A_3*X - b_cell)' * (A_3*X - b_cell) + r2 * X' * D3 * X);
x_cell_2_3 = fmincon(fun_cell_23, Init_3, Aineq_3, bineq_3, Aeq, beq, lb, ub, nonlcon, options);
x_cell_23=x_cell_2_3;
x_cell_2_3(x_cell_2_3 < 0.001) = 0;

fun_cell_31 = @(X) ((A_1*X - b_cell)' * (A_1*X - b_cell) + r3 * X' * D1 * X);
x_cell_3_1 = fmincon(fun_cell_31, Init_1, Aineq_1, bineq_1, Aeq, beq, lb, ub, nonlcon, options);
x_cell_31=x_cell_3_1;
x_cell_3_1(x_cell_3_1 < 0.001) = 0;

b_DRT_11 = A_1 * x_cell_11(:, 1);
Z_real_DRT_11 = b_DRT_11(1:N_mea, 1);
Z_imag_DRT_11 = b_DRT_11(N_mea+1:2*N_mea, 1);

b_DRT_21 = A_1 * x_cell_21(:, 1);
Z_real_DRT_21 = b_DRT_21(1:N_mea, 1);
Z_imag_DRT_21 = b_DRT_21(N_mea+1:2*N_mea, 1);

b_DRT_22 = A_2 * x_cell_22(:, 1);
Z_real_DRT_22 = b_DRT_22(1:N_mea, 1);
Z_imag_DRT_22 = b_DRT_22(N_mea+1:2*N_mea, 1);

b_DRT_23 = A_3 * x_cell_23(:, 1);
Z_real_DRT_23 = b_DRT_23(1:N_mea, 1);
Z_imag_DRT_23 = b_DRT_23(N_mea+1:2*N_mea, 1);

b_DRT_31 = A_1 * x_cell_31(:, 1);
Z_real_DRT_31 = b_DRT_31(1:N_mea, 1);
Z_imag_DRT_31 = b_DRT_31(N_mea+1:2*N_mea, 1);

figure(1)
hold on
idx1 = x_cell_1_1(1:N_rq_1,1) > 0;
x_vals_1 = x_rq_1(idx1,1);
y_vals_1 = x_cell_1_1(idx1,1);
plot(x_vals_1, y_vals_1, 'or','DisplayName', '\lambda_T=1','MarkerFaceColor', 'r','LineWidth', 1);
for i = 1:length(x_vals_1)
line([x_vals_1(i), x_vals_1(i)], [0, y_vals_1(i)], 'Color', 'r', 'LineStyle', '-', 'LineWidth', 1, 'HandleVisibility', 'off');
end

idx2 = x_cell_2_1(1:N_rq_1,1) > 0;
x_vals_2 = x_rq_1(idx2,1);
y_vals_2 = x_cell_2_1(idx2,1);
plot(x_vals_2, y_vals_2, 'ob', 'DisplayName', '\lambda_T=0.1','MarkerFaceColor', 'b','LineWidth', 1);
for i = 1:length(x_vals_2)
line([x_vals_2(i), x_vals_2(i)], [0, y_vals_2(i)], 'Color', 'b', 'LineStyle', '-', 'LineWidth', 1, 'HandleVisibility', 'off');
end

idx3 = x_cell_3_1(1:N_rq_1,1) > 0;
x_vals_3 = x_rq_1(idx3,1);
y_vals_3 = x_cell_3_1(idx3,1);
plot(x_vals_3, y_vals_3, 'og', 'DisplayName', '\lambda_T=0.01','MarkerFaceColor', 'g','LineWidth', 1);
for i = 1:length(x_vals_3)
line([x_vals_3(i), x_vals_3(i)], [0, y_vals_3(i)], 'Color', 'g', 'LineStyle', '-', 'LineWidth', 1, 'HandleVisibility', 'off');
end
font_name = 'Times New Roman';
font_size = 15;
legend('show', 'FontName', font_name, 'FontSize', font_size);
xlabel('lg(\tau)', 'FontName', font_name, 'FontSize', font_size);
ylabel('\gamma(\tau)', 'FontName', font_name, 'FontSize', font_size);
set(gca, 'FontSize', font_size);
set(gca, 'LineWidth', 2);
box on
x_range = [1.5, 4];
y_range = [0, 0.006];
rectangle('Position', [x_range(1), y_range(1), diff(x_range), diff(y_range)],'EdgeColor', [0.5 0.5 0.5], 'LineStyle', '--', 'LineWidth', 1.5, 'FaceColor', [0.8 0.8 0.8 0.3]); 
ax_inset_1 = axes('Position', [0.425 0.275 0.45 0.30]);  
box on;
hold on;
plot(x_vals_1, y_vals_1, 'or','DisplayName', '\lambda_T=1','MarkerFaceColor', 'r','LineWidth', 1);
for i = 1:length(x_vals_1)
line([x_vals_1(i), x_vals_1(i)], [0, y_vals_1(i)], 'Color', 'r', 'LineStyle', '-', 'LineWidth', 1, 'HandleVisibility', 'off');
end
plot(x_vals_2, y_vals_2, 'ob', 'DisplayName', '\lambda_T=0.1','MarkerFaceColor', 'b','LineWidth', 1);
for i = 1:length(x_vals_2)
line([x_vals_2(i), x_vals_2(i)], [0, y_vals_2(i)], 'Color', 'b', 'LineStyle', '-', 'LineWidth', 1, 'HandleVisibility', 'off');
end
plot(x_vals_3, y_vals_3, 'og', 'DisplayName', '\lambda_T=0.01','MarkerFaceColor', 'g','LineWidth', 1);
for i = 1:length(x_vals_3)
line([x_vals_3(i), x_vals_3(i)], [0, y_vals_3(i)], 'Color', 'g', 'LineStyle', '-', 'LineWidth', 1, 'HandleVisibility', 'off');
end
xlim(x_range);
ylim(y_range);
set(gca, 'FontSize', 10);

figure(2)
hold on
plot(x_vals_2, y_vals_2, 'or', 'DisplayName', 'N_D_R_T=50','MarkerFaceColor', 'r','LineWidth', 1);
for i = 1:length(x_vals_2)
line([x_vals_2(i), x_vals_2(i)], [0, y_vals_2(i)], 'Color', 'r', 'LineStyle', '-', 'LineWidth', 1, 'HandleVisibility', 'off');
end
idx4 = x_cell_2_2(1:N_rq_2,1) > 0;
x_vals_4 = x_rq_2(idx4,1);
y_vals_4 = x_cell_2_2(idx4,1);
plot(x_vals_4, y_vals_4, 'ob', 'DisplayName', 'N_D_R_T=100','MarkerFaceColor', 'b','LineWidth', 1);
for i = 1:length(x_vals_4)
line([x_vals_4(i), x_vals_4(i)], [0, y_vals_4(i)], 'Color', 'b', 'LineStyle', '-', 'LineWidth', 1, 'HandleVisibility', 'off');
end
idx5 = x_cell_2_3(1:N_rq_3,1) > 0;
x_vals_5 = x_rq_3(idx5,1);
y_vals_5 = x_cell_2_3(idx5,1);
plot(x_vals_5, y_vals_5, 'og', 'DisplayName', 'N_D_R_T=200','MarkerFaceColor', 'g','LineWidth', 1);
for i = 1:length(x_vals_5)
line([x_vals_5(i), x_vals_5(i)], [0, y_vals_5(i)], 'Color', 'g', 'LineStyle', '-', 'LineWidth', 1, 'HandleVisibility', 'off');
end
legend('show', 'FontName', font_name, 'FontSize', font_size);
xlabel('lg(\tau)', 'FontName', font_name, 'FontSize', font_size);
ylabel('\gamma(\tau)', 'FontName', font_name, 'FontSize', font_size);
set(gca, 'FontSize', font_size);
set(gca, 'LineWidth', 2);
box on
x_range = [1.5, 4];
y_range = [0, 0.006];
rectangle('Position', [x_range(1), y_range(1), diff(x_range), diff(y_range)],'EdgeColor', [0.5 0.5 0.5], 'LineStyle', '--', 'LineWidth', 1.5, 'FaceColor', [0.8 0.8 0.8 0.3]); 
ax_inset_1 = axes('Position', [0.425 0.275 0.45 0.30]);  
box on;
hold on;
plot(x_vals_2, y_vals_2, 'or', 'DisplayName', 'N_D_R_T=50','MarkerFaceColor', 'r','LineWidth', 1);
for i = 1:length(x_vals_2)
line([x_vals_2(i), x_vals_2(i)], [0, y_vals_2(i)], 'Color', 'r', 'LineStyle', '-', 'LineWidth', 1, 'HandleVisibility', 'off');
end
plot(x_vals_4, y_vals_4, 'ob', 'DisplayName', 'N_D_R_T=100','MarkerFaceColor', 'b','LineWidth', 1);
for i = 1:length(x_vals_4)
line([x_vals_4(i), x_vals_4(i)], [0, y_vals_4(i)], 'Color', 'b', 'LineStyle', '-', 'LineWidth', 1, 'HandleVisibility', 'off');
end
plot(x_vals_5, y_vals_5, 'og', 'DisplayName', 'N_D_R_T=200','MarkerFaceColor', 'g','LineWidth', 1);
for i = 1:length(x_vals_5)
line([x_vals_5(i), x_vals_5(i)], [0, y_vals_5(i)], 'Color', 'g', 'LineStyle', '-', 'LineWidth', 1, 'HandleVisibility', 'off');
end
xlim(x_range);
ylim(y_range);
set(gca, 'FontSize', 10);

figure(3);
hold on;
plot(Z_real_DRT_11, -Z_imag_DRT_11, '-','LineWidth', 1, 'MarkerSize', 8, 'Color', 'k','DisplayName', 'lambda_T=0.5,N_r_q=50', 'Marker', 'o', 'MarkerFaceColor', 'none');
plot(Z_real_DRT_21, -Z_imag_DRT_21, '-', 'LineWidth', 1, 'MarkerSize', 8, 'Color', 'r','DisplayName', 'lambda_T=0.1,N_r_q=50', 'Marker', 'o', 'MarkerFaceColor', 'none');
plot(Z_real_DRT_22, -Z_imag_DRT_22, '-', 'LineWidth', 1, 'MarkerSize', 8, 'Color', 'b', 'DisplayName', 'lambda_T=0.1,N_r_q=100', 'Marker', 'o', 'MarkerFaceColor', 'none');
plot(Z_real_DRT_23, -Z_imag_DRT_23, '-', 'LineWidth', 1, 'MarkerSize', 8, 'Color', 'g', 'DisplayName', 'lambda_T=0.1,N_r_q=200', 'Marker', 'o', 'MarkerFaceColor', 'none');
plot(Z_real_DRT_31, -Z_imag_DRT_31, '-', 'LineWidth', 1, 'MarkerSize', 8, 'Color', 'c', 'DisplayName', 'lambda_T=0.01,N_r_q=50', 'Marker', 'o', 'MarkerFaceColor', 'none');
plot(Z_real, -Z_imag, '-oy', 'LineWidth', 1.5, 'MarkerSize', 3,'DisplayName', 'Experiment');
legend('show', 'FontName', 'Times New Roman', 'FontSize', 30);
xlabel('Real(Z) (\Omega)', 'FontName', 'Times New Roman', 'FontSize', 30);                          
ylabel('-Imag(Z) (\Omega)', 'FontName', 'Times New Roman', 'FontSize', 30);
grid on;
legend('show', 'FontName', font_name, 'FontSize', font_size);
set(gca, 'FontSize', font_size);
box on
hold off;

function D = createM(M1)
A1=zeros(M1);
A2=ones(M1);
A11=zeros(2*M1);
A12=ones(2*M1);
A3=zeros(2*M1,2);
A4=zeros(2,2*M1);
A5=zeros(2,2);
for m=1:1:M1
    for n=1:1:M1
        if m<=n
                A2(m,n)=0;
        end
    end
end
for m=1:1:2*M1
    for n=1:1:2*M1
        if m<=n
                A12(m,n)=0;
        end
    end
end
D1=[A2,A1;A1,A2];
D2=[A1,A1;A1,A1];
D ={ D1   D2   A11  A11  D2   A3;
     D2   D1   A11  A11  D2   A3;
     A11  A11  A12  A11  D2   A3;
     A11  A11  A11  A12  D2   A3;
     D2   D2   D2   D2   D1   A3;
     A4   A4   A4   A4   A4   A5};
D=cell2mat(D);
end

function A=createAM(N_rq,wi,w,N)
n=zeros(N_rq+1+1,1);
n(1:N_rq)=0.95;
A=zeros(2*N,N_rq+1+1);
for i=1:1:N
    for j=1:1:N_rq
        A(i,j)=(1+(w(i)/wi(j))^n(j)*cos(n(j)*pi/2))/(1+2*(w(i)/wi(j))^n(j)*cos(n(j)*pi/2)+(w(i)/wi(j))^(2*n(j)));
        
    end    
    A(i,N_rq+1)=1;
    A(i,N_rq+2)=0; 
end
for i=N+1:1:2*N
    for j=1:1:N_rq
        A(i,j)=-(w(i-N)/wi(j))^n(j)*sin(n(j)*pi/2)/(1+2*(w(i-N)/wi(j))^n(j)*cos(n(j)*pi/2)+(w(i-N)/wi(j))^(2*n(j)));
    end
    A(i,N_rq+1)=0;
    A(i,N_rq+2)=1*w(i-N);
end
end

function [Aineq, bineq]=createAb(N_rq)
Aineq=zeros(N_rq+1+1,N_rq+1+1);
for i=1:1:N_rq+1+1
    Aineq(i,i)=-1;
end
bineq=zeros(N_rq+1+1,1);
end