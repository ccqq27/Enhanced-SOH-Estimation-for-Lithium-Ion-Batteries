clc;
clear;
%% LOAD DATA
load data.mat
%% Data preprocessing
data_EIS=SOC_55_80cyc;
N_mea=length(data_EIS(:,1));
f_mea=data_EIS(:,1);
w_mea=2*pi*f_mea;
Z_real=data_EIS(:,2);
Z_imag=data_EIS(:,3);
%% DRT preset value
RC_delta_1=0.2; 
RC_delta_2=0.1; 
RC_delta_3=0.05; 
RC_max=6;
RC_min=-6;     
RL_delta=0.05; 
r1=0.5;     
r2=0.1;     
r3=0.01;     
%% Calculation of the initial parameters of DRT
N_rc_1=(RC_max-RC_min)/RC_delta_1;
N_rc_2=(RC_max-RC_min)/RC_delta_2;
N_rc_3=(RC_max-RC_min)/RC_delta_3;
f_rc_1=10.^[RC_min:RC_delta_1:RC_max-RC_delta_1];
f_rc_2=10.^[RC_min:RC_delta_2:RC_max-RC_delta_2];
f_rc_3=10.^[RC_min:RC_delta_3:RC_max-RC_delta_3];
w_rc_1=2*pi*f_rc_1;
w_rc_2=2*pi*f_rc_2;
w_rc_3=2*pi*f_rc_3;
N_rl=1;
Init_1=0.00002*ones(N_rc_1+N_rl+1,1);
Init_2=0.00002*ones(N_rc_2+N_rl+1,1);
Init_3=0.00002*ones(N_rc_3+N_rl+1,1);
%% DRT constructs the A matrix
A_1 = createAM(N_rc_1, w_rc_1, w_mea, N_mea);
A_2 = createAM(N_rc_2, w_rc_2, w_mea, N_mea);
A_3 = createAM(N_rc_3, w_rc_3, w_mea, N_mea);
%% Construction of DRT_b matrix
b_cell=zeros(2*N_mea,1);
for m=1:1:N_mea
    b_cell(m,1)=Z_real(m);
end
for m=N_mea+1:1:2*N_mea
    b_cell(m,1)=Z_imag(m-N_mea);
end
%% The solution Settings for the least squares solution of DRT
[Aineq_1, bineq_1]=createAb(N_rc_1);
[Aineq_2, bineq_2]=createAb(N_rc_2);
[Aineq_3, bineq_3]=createAb(N_rc_3);
Aeq=[];
beq=[];
options = optimoptions('fmincon','Algorithm','sqp');
nonlcon = [];
lb = [];
ub = [];
%% DRT regularization
D1 = createM(N_rc_1);
D2 = createM(N_rc_2);
D3 = createM(N_rc_3);
%% DRT solution
fun_cell_11=@(X)((A_1*X-b_cell)'*(A_1*X-b_cell)+r1*(D1*X)'*(D1*X));
x_cell_11 = fmincon(fun_cell_11,Init_1,Aineq_1,bineq_1,Aeq,beq,lb,ub,nonlcon,options);

fun_cell_21=@(X)((A_1*X-b_cell)'*(A_1*X-b_cell)+r2*(D1*X)'*(D1*X));
x_cell_21 = fmincon(fun_cell_21,Init_1,Aineq_1,bineq_1,Aeq,beq,lb,ub,nonlcon,options);

fun_cell_22=@(X)((A_2*X-b_cell)'*(A_2*X-b_cell)+r2*(D2*X)'*(D2*X));
x_cell_22 = fmincon(fun_cell_22,Init_2,Aineq_2,bineq_2,Aeq,beq,lb,ub,nonlcon,options);

fun_cell_23=@(X)((A_3*X-b_cell)'*(A_3*X-b_cell)+r2*(D3*X)'*(D3*X));
x_cell_23 = fmincon(fun_cell_23,Init_3,Aineq_3,bineq_3,Aeq,beq,lb,ub,nonlcon,options);

fun_cell_31=@(X)((A_1*X-b_cell)'*(A_1*X-b_cell)+r3*(D1*X)'*(D1*X));
x_cell_31 = fmincon(fun_cell_31,Init_1,Aineq_1,bineq_1,Aeq,beq,lb,ub,nonlcon,options);

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

x_rc_1=(-log10(w_rc_1))';
x_rc_2=(-log10(w_rc_2))';
x_rc_3=(-log10(w_rc_3))';

figure(1);
hold on;
plot(x_rc_1(1:N_rc_1,1), x_cell_11(1:N_rc_1,1), '-or', 'DisplayName', '\lambda_T=0.5', 'MarkerFaceColor', 'r', 'LineWidth', 1);
plot(x_rc_1(1:N_rc_1,1), x_cell_21(1:N_rc_1,1), '-ob', 'DisplayName', '\lambda_T=0.1', 'MarkerFaceColor', 'b', 'LineWidth', 1);
plot(x_rc_1(1:N_rc_1,1), x_cell_31(1:N_rc_1,1), '-og', 'DisplayName', '\lambda_T=0.01', 'MarkerFaceColor', 'g', 'LineWidth', 1);
font_name = 'Times New Roman';
font_size = 15;
legend('show', 'FontName', font_name, 'FontSize', font_size);
xlabel('lg(\tau)', 'FontName', font_name, 'FontSize', font_size);
ylabel('\gamma(\tau)', 'FontName', font_name, 'FontSize', font_size);
set(gca, 'FontSize', font_size);
set(gca, 'LineWidth', 2);
box on;
x_range = [-5, 2];
y_range = [0, 0.002];
rectangle('Position', [x_range(1), y_range(1), diff(x_range), diff(y_range)],'EdgeColor', [0.5 0.5 0.5], 'LineStyle', '--', 'LineWidth', 1.5, 'FaceColor', [0.8 0.8 0.8 0.3]); 
ax_inset_1 = axes('Position', [0.20 0.40 0.35 0.45]);  
box on;
hold on;
plot(x_rc_1(1:N_rc_1,1), x_cell_11(1:N_rc_1,1), '-or', 'MarkerFaceColor', 'r', 'LineWidth', 1);
plot(x_rc_1(1:N_rc_1,1), x_cell_21(1:N_rc_1,1), '-ob', 'MarkerFaceColor', 'b', 'LineWidth', 1);
plot(x_rc_1(1:N_rc_1,1), x_cell_31(1:N_rc_1,1), '-og', 'MarkerFaceColor', 'g', 'LineWidth', 1);
xlim(x_range);
ylim(y_range);
set(gca, 'FontSize', 10);

figure(2);
hold on;
plot(x_rc_1(1:N_rc_1,1),x_cell_21(1:N_rc_1,1),'-or','DisplayName', 'N_D_R_T=50','MarkerFaceColor', 'r','LineWidth', 1);
plot(x_rc_2(1:N_rc_2,1),x_cell_22(1:N_rc_2,1),'-ob','DisplayName', 'N_D_R_T=100','MarkerFaceColor', 'b','LineWidth', 1);
plot(x_rc_3(1:N_rc_3,1),x_cell_23(1:N_rc_3,1),'-og','DisplayName', 'N_D_R_T=200','MarkerFaceColor', 'g','LineWidth', 1);
font_name = 'Times New Roman';
font_size = 15;
legend('show', 'FontName', font_name, 'FontSize', font_size);
xlabel('lg(\tau)', 'FontName', font_name, 'FontSize', font_size);
ylabel('\gamma(\tau)', 'FontName', font_name, 'FontSize', font_size);
set(gca, 'FontSize', font_size);
set(gca, 'LineWidth', 2);
box on;
x_range = [-5, 2];
y_range = [0, 0.002];
rectangle('Position', [x_range(1), y_range(1), diff(x_range), diff(y_range)],'EdgeColor', [0.5 0.5 0.5], 'LineStyle', '--', 'LineWidth', 1.5, 'FaceColor', [0.8 0.8 0.8 0.3]); 
ax_inset_2 = axes('Position', [0.20 0.40 0.35 0.45]);  
box on;
hold on;
plot(x_rc_1(1:N_rc_1,1),x_cell_21(1:N_rc_1,1),'-or','MarkerFaceColor', 'r','LineWidth', 1);
plot(x_rc_2(1:N_rc_2,1),x_cell_22(1:N_rc_2,1),'-ob','MarkerFaceColor', 'b','LineWidth', 1);
plot(x_rc_3(1:N_rc_3,1),x_cell_23(1:N_rc_3,1),'-og','MarkerFaceColor', 'g','LineWidth', 1);
xlim(x_range);
ylim(y_range);
set(gca, 'FontSize', 10);


figure(3);
hold on;
plot(Z_real_DRT_11, -Z_imag_DRT_11, '-','LineWidth', 1, 'MarkerSize', 8, 'Color', 'k','DisplayName', '\lambda_T=0.5,N_r_c=50', 'Marker', 'o', 'MarkerFaceColor', 'none');
plot(Z_real_DRT_21, -Z_imag_DRT_21, '-', 'LineWidth', 1, 'MarkerSize', 8, 'Color', 'r','DisplayName', '\lambda_T=0.1,N_r_c=50', 'Marker', 'o', 'MarkerFaceColor', 'none');
plot(Z_real_DRT_22, -Z_imag_DRT_22, '-', 'LineWidth', 1, 'MarkerSize', 8, 'Color', 'b', 'DisplayName', '\lambda_T=0.1,N_r_c=100', 'Marker', 'o', 'MarkerFaceColor', 'none');
plot(Z_real_DRT_23, -Z_imag_DRT_23, '-', 'LineWidth', 1, 'MarkerSize', 8, 'Color', 'g', 'DisplayName', '\lambda_T=0.1,N_r_c=200', 'Marker', 'o', 'MarkerFaceColor', 'none');
plot(Z_real_DRT_31, -Z_imag_DRT_31, '-', 'LineWidth', 1, 'MarkerSize', 8, 'Color', 'c', 'DisplayName', '\lambda_T=0.01,N_r_c=50', 'Marker', 'o', 'MarkerFaceColor', 'none');
plot(Z_real, -Z_imag, '-oy', 'LineWidth', 1.5, 'MarkerSize', 3,'DisplayName', 'Experiment');
legend('show', 'FontName', 'Times New Roman', 'FontSize', font_size);
xlabel('Real(Z) (\Omega)', 'FontName', 'Times New Roman', 'FontSize', font_size);                          
ylabel('-Imag(Z) (\Omega)', 'FontName', 'Times New Roman', 'FontSize', font_size);
xlim([0.016 0.034]);    
ylim([-0.006 0.011]);  
grid on;
legend('show', 'FontName', font_name, 'FontSize', font_size/2);
set(gca, 'FontSize', font_size);
box on
hold off;

function A=createAM(N_rc,w_rc,w_mea,N_mea)
    A=zeros(2*N_mea,N_rc+1+1);
    for i=1:1:N_mea
        for j=1:1:N_rc
            A(i,j)=1/(1+(w_mea(i)/w_rc(j))^2);
        end    
        A(i,N_rc+1)=1;
        A(i,N_rc+2)=0;    
    end
    for i=N_mea+1:1:2*N_mea
        for j=1:1:N_rc
            A(i,j)=-(w_mea(i-N_mea)/w_rc(j))/(1+(w_mea(i-N_mea)/w_rc(j))^2);
        end
        A(i,N_rc+1)=0;
        A(i,N_rc+2)=1*w_mea(i-N_mea);
    end
end
function [Aineq, bineq]=createAb(N_rc)
Aineq=zeros(N_rc+1+1,N_rc+1+1);
for i=1:1:N_rc+1+1
    Aineq(i,i)=-1;
end
bineq=zeros(N_rc+1+1,1);
end
function D1 = createM(N_rc)
D1=zeros(N_rc+1+1,N_rc+1+1);
for n=2:1:N_rc-1
    D1(n,n-1)=-1;
    D1(n,n)=2;
    D1(n,n+1)=-1;
end
D1(1,1)=1;
D1(1,2)=-1;
D1(N_rc,N_rc-1)=-1;
D1(N_rc,N_rc)=1;
end