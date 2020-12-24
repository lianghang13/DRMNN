clear all
load virus_drug_matrix;
load virussim_matrix;
load drugsim_matrix;
maxiter = 300;
alpha = 100;
beta = 100;
tol1 = 2*1e-3;
tol2 = 1*1e-5;  
[dn,dr] = size(virusdrug);
Vp=find(virusdrug()==1);
Vn=find(virusdrug()==0);
MatPredict=zeros(205,34);
Ip=crossvalind('Kfold',numel(Vp),5);
In=crossvalind('Kfold',numel(Vn),5);
for I=1:5
    vp=Ip==I;
    vn=In==I;
    matDT=virusdrug;
    matDT(Vp(vp))=0;
    T = [virussim, matDT'; matDT, drugsim];
    [t1, t2] = size(T);
    trIndex = double(T ~= 0);
    [WW,iter] = DRMNN(alpha, beta, T, trIndex, tol1, tol2, maxiter, 0, 1);
    recMatrix = WW((t1-dn+1) : t1, 1 : dr);
    V=[Vn(vn);Vp(vp)];
    MatPredict(V)=recMatrix(V);
end
   [AUC,AUPR,Acc,Sen,Spe,Pre]=ROCcompute(MatPredict(),virusdrug(),1);  

 
