                                 %T A K I N G  I N P U T

imcount=20;
nps=2;
%image dimensions = 120X128
size = 120*128;
%size of coloumn vector to be 1080000
input = zeros(size,imcount);
input=double(input);
for i=1:imcount
    f=imread(strcat(int2str(i),'.pgm'));
   % f=rgb2gray(f);
    f=double(f);
    f=reshape(f.',[],1);
    input(:,i)=f;
end
                                % MEAN OF INPUT IMAGE
                    
in_mean = (mean(input.'))';

                             %CALCULATING ZERO MEAN MATRIX
                    
zero_mean=double(zeros(size,imcount));
for i=1:imcount
    zero_mean(:,i)=input(:,i)-in_mean(:,1);
end

                              %SURROGATE COVARIANCE
                    
cov = zero_mean' * zero_mean;

                            %EigenValue and Eigen vector
                      
[Evec,Eval] = eig(cov);

Eval = diag(Eval);
Evalue= sort(Eval,'descend');

                    %Selecting top eigen values by setting k
                    
k=9;
topEval=Evalue(1:k);

                          %Calculating Feature Vector
                        
position =[];
for i=1:k;
    for j=1:imcount
        if Eval(j)== topEval(i)
            position(i)=j;
            break;
        end
    end
end
    
featureVec =double(zeros(imcount,k));
for i=1:k
    featureVec(:,i) = Evec(:,position(i));
end

                            %Calculating EigenFace and EigenSignature
         
Eigenface= featureVec' * zero_mean';
EigenSign = Eigenface * zero_mean;
nC=imcount/2;                %Finding number of classes
i=1;
j=1;
                             %Finding mean of each class
CMean =double(zeros(k,nC));  
while i < imcount+1
         CMean(:,j)=(EigenSign(:,i)+EigenSign(:,i+1))/2;
         j=j+1;
         i=i+2;
end       
                          %calculating mean of ProjectedFaces
ProMean = (mean(EigenSign.'))';

                           %Calculating within Class Scatter
SW =double(zeros(k,k));


