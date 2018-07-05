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
for i=1:k
mat=uint8(vec2mat(Eigenface(i,:),128));
subplot(3,3,i);
imshow(mat);
end
EigenSign = Eigenface * zero_mean;
nC=imcount/2;                %Finding number of classes
i=1;
j=1;
                             %Finding mean of each class
CMean =double(zeros(k,nC));  
while i <=imcount
         CMean(:,j)=(EigenSign(:,i)+EigenSign(:,i+1))/2;
         j=j+1;
         i=i+2;
end       
                          %calculating mean of ProjectedFaces
ProMean = (mean(EigenSign.'))';

                           %Calculating within Class Scatter
SW =double(zeros(k,k));
j=1;
i=1;
while i <=imcount
    XX=(EigenSign(:,i)- CMean(:,j));
    YY=(EigenSign(:,i+1)- CMean(:,j));
    SW=SW+(XX*XX')+(YY*YY');
    j=j+1;
    i=i+2;
end





                           %Calculating Between Class Scatter
SB =double(zeros(k,k));
for i=1:nC
    D = CMean(:,i);
    SB=SB+((CMean(:,i)-ProMean(:,1))*D');
end
                            %Calculating criterion Function
J= inv(SW)*SB;
[evec,eval] = eig(J);       %calculating eigen values and vector
eval = diag(eval);
evalue= sort(eval,'descend');
m=6;
topeval=evalue(1:m);        %selecting top eigen values
    pos =[];
for i=1:m;
    for j=1:imcount
        if eval(j)== topeval(i)
            pos(i)=j;
            break;
        end
    end
end
                        %constructing new feature matrix
                        
W =double(zeros(k,m));
for i=1:m
    W(:,i) = evec(:,pos(i));
end
                        %Finding Fischer Faces
FisherFace= (W')*EigenSign;

                                    %TESTING
                             
                                    
test_im=5; %input test image no.
test_im = imread(strcat(int2str(test_im),'k.pgm'));
%test_im=rgb2gray(test_im);
test_im=double(test_im);
test_im=reshape(test_im.',[],1);
zero_test_im = test_im - in_mean;
                        %calculating Projected Eigen Face
pro_im= Eigenface * zero_test_im;
                          %Calculating Fischer Test img
FishPro=(W')*pro_im;
 
                         %calculating Euclidean Distance
eucDist= [];
for i=1:imcount
    eucDist(i)= norm(FisherFace(:,i)-FishPro(:,1));
end
[M,I] = min(eucDist')
out=imread(strcat(int2str(I),'.pgm'));
figure;
imshow(out);