function im=gradT(gim)

[ny,nx,m]=size(gim);


div1=zeros(ny,nx);
div2=zeros(ny,nx);

div1(:,1)=gim(:,1,1);
div1(:,end)=-gim(:,end-1,1);
div1(:,2:end-1)=gim(:,2:end-1,1)-gim(:,1:end-2,1);

div2(1,:)=gim(1,:,2);
div2(end,:)=-gim(end-1,:,2);
div2(2:end-1,:)=gim(2:end-1,:,2)-gim(1:end-2,:,2);

im=-(div1+div2);