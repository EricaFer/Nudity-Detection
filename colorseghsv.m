function segmap=colorseghsv(rgbImage)
hsv=rgb2hsv(rgbImage);
      colors={'green','rose','blue','yellow','green'};
      BW=false;
      for i=1:length(colors)
        BW=BW|markerbin(colors{i},hsv);
      end    
BW = imclose(BW,ones(5));
for ii=3:-1:1
   Q=regionprops(BW,rgbImage(:,:,ii),'MeanIntensity');
     cmap(:,ii)=vertcat(Q.MeanIntensity)/255;
end
segmap=label2rgb(bwlabel(BW),cmap,'k');
function bw=markerbin(label,hsv)
    vthresh=.9; 
   H=hsv(:,:,1);
   S=hsv(:,:,2);
   V=hsv(:,:,3);
     Vbin=V>=vthresh;
   switch label
       case 'blue'
           Hbin=abs(H-.55)<=.1;
            Sbin=S>=.5;
       case 'yellow'
           Hbin=abs(H-.16)<=.1;
            Sbin=S>=.5;
       case 'green'
           Hbin=abs(H-.325)<=.1;
           Sbin=S>=.5;
       case 'orange'    
           Hbin=abs(H-.125)<=.025;
           Sbin=S>=.5; 
       case 'rose' 
           Hbin=abs(H-.85)<=.1; 
           Sbin=S>=.5;
   end
    bw=Hbin&Sbin&Vbin;