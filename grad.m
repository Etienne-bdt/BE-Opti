function gim=grad(im)

[ny,nx]=size(im);
gim=zeros(ny,nx,2);

gim(:,1:end-1,1)=im(:,2:end)-im(:,1:end-1);
gim(1:end-1,:,2)=im(2:end,:)-im(1:end-1,:);