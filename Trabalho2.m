im = imread("hommeditpatrocle.jpg");
[H,S,V] = rgb2hsv(im);
figure, imshow(HSV);




rgbImage = imread("hommeditpatrocle.jpg");
subplot(1,2,1);
imshow(rgbImage);
title('Original Image');
redChannel=rgbImage(:, :, 1);
greenChannel=rgbImage(:, :, 2);
blueChannel=rgbImage(:, :, 3);
data=double([redChannel(:), greenChannel(:), blueChannel(:)]);
for i=1:10
numberOfClasses=i;
[m, n]=kmeans(data,numberOfClasses);
m=reshape(m,size(rgbImage,1),size(rgbImage,2));
n=n/255;
clusteredImage=label2rgb(m,n);
subplot(1,2,2);
imshow(clusteredImage);
title(i);
pause;
end

lab_he = rgb2lab([H,S]);
figure, imshow(lab_he), title('rgb2lab');