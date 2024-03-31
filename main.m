close all

load('image1.mat');
load('motion_blur.mat');

imdisp(x_star, 'Image x');
imdisp(fftshift(h), 'Noyau h');

%% Génération de y

x_fft = fft2(x_star);
h_fft = fft2(h);

y_fft = x_fft .* h_fft;

y= ifft2(y_fft);

imdisp(y, 'Image y');

%% Partie 2

% L'opérateur se résume comme ca :
H_star = @(x) ifft2(conj(h_fft).*fft2(x));
HcH = @(x) ifft2(fft2(x).\abs(h_fft).^2);
%H = F-1DF
%FH = DF
%FHF-1 = D

% la condition d'optimalité, est H_star * H * x = H_star*y 
%H_s = (FsDsF)
%H_sH = FsDsFFsDF = Fs(DsD)F
%(H_sH)-1 = Fs-1(DsD)-1F-1
%H_sH -1 = F-1 (DsD)-1 Fs-1


%(H_sH)-1Hs = F-1(DsD)-1DsF = F-1 (D-1 Ds-1 Ds F) = F-1 D-1 F




x_ch = ifft2((abs(h_fft)).^(-2).*conj(h_fft).*y_fft);

imdisp(x_ch,'x chapeau')

%% 5 girafe

load('image2_bruit.mat')

imdisp(y_bruit,'Image flou + bruit')

%mm le bruit est pas visible, girafe 

x_ch2 = ifft2((abs(h_fft)).^(-2).*conj(h_fft).*fft2(y_bruit));

imdisp(x_ch2, 'Image débruitée');

%% Partie 3 : Gradients

g = grad(x_star);
imdisp(g(:,:,1),'Gradient horizontal');
imdisp(g(:,:,2),'Gradient Vertical');


%% algo
%GT Gx 
%grad f = HsHx-Hsy + lambda/2 * GTGx

lambda = 1e-1;
%gradf = @(x) Hs(Hx - y) + lambda * gradT(grad(x)); 
% Hs = FsDsF
% HsH = Fs(DsD)F => 




gradf = @(x) ifft2((abs(h_fft).^2.*fft2(x)) - conj(h_fft).*fft2(y_bruit)) + lambda * gradT(grad(x));
f = @(x) 0.5*(norm(h_fft.*fft2(x)- fft2(y_bruit),'fro')^2 + (lambda)* sum(grad(x).^2,'all'));

k = 1;
kmax = 1000;
x= zeros(size(y));
z = zeros(size(y));
fval = zeros(1,kmax);
s = 1/(max(max(abs(h_fft)))^2+(lambda^2)*8); 
while k<kmax
    xp=x;
    x = z - s*gradf(z);
    z = x + ((k-1)/(k+2)) *(x-xp); 
    fval(k) = f(z);
    k= k+1;
end
figure;
semilogy(1:kmax,fval);
imdisp(z,'Image z')

%% ADMM

ProxL1 = @(x,gamma) sign(x).*max(0,abs(x)-gamma); 

load fourierGtG.mat

xadmm = zeros(size(y_bruit));
z = xadmm;
k2=1;
k2max = 100;
lambda_i=0.1;
fADMM = @(x) 0.5*(norm(h_fft.*fft2(x)- fft2(y_bruit),'fro')^2) + (lambda_i/2)* sum(abs(grad(x)),'all');
A=grad(xadmm);
rho = 0.00001;
lambda= lambda_i;
fval2 = zeros(1,k2max);
while k2<k2max
% A = grad ,rho arbitraire positif
    V = ProxL1(A+lambda/rho,1/rho);
    xadmm = abs(ifft2(1./(abs(h_fft).^2+rho.*Fgtg).*conj(h_fft).*fft2(y_bruit)) + rho * gradT(lambda/rho-V));
    A=grad(xadmm);
    lambda = lambda + rho*(A-V);
%   
    fval2(k2) = fADMM(real(xadmm));
% Pour x Condition d'optimalite -> sys lineaire -> grad = 0 donc GtG -> HtH + GtG

   
%HsHx - Hsy + GtGx = 0
    % x = (HsH)-1 Hsy - la/2(HsH)-1GtGx

% lambda    
    % la condition d'optimalité, est H_star * H * x = H_star*y - lambda/2
    % GtG x

%H_s = (FsDsF)
%H_sH = FsDsFFsDF = Fs(DsD)F
%(H_sH)-1 = Fs-1(DsD)-1F-1
%H_sH -1 = F-1 (DsD)-1 Fs-1
    % (HsH + GtG)x = HsY -> (HsH)-1Hs Y + (GtG)-1Y
    % x = F-1 DsD-1 F Y + F-1D2FY
%d'ou x = F-1DF*y - lambda/2 F-1 (DsD)-1Fgtg F

    k2=k2+1;
end
figure;
semilogy(1:k2max,fval2);
imdisp(xadmm, 'X ADMM')
