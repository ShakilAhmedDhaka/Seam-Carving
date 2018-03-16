#include <opencv2/opencv.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace std;

Mat equalizeIntensity(const Mat& inputImage)
{
    if(inputImage.channels() >= 3)
    {
        Mat ycrcb;

        cvtColor(inputImage,ycrcb,CV_BGR2YCrCb);

        vector<Mat> channels;
        split(ycrcb,channels);

        equalizeHist(channels[0], channels[0]);

        Mat result;
        merge(channels,ycrcb);

        cvtColor(ycrcb,result,CV_YCrCb2BGR);

        return result;
    }
    return Mat();
}

bool energy_map(Mat& src, Mat& grad){

  int scale = 1;
  int delta = 0;
  int ddepth = CV_16S;

  Mat grad_x, grad_y;
  Mat abs_grad_x, abs_grad_y;
  Mat dst = src.clone();
  // GaussianBlur( src, dst, Size(11,11), 0, 0, BORDER_DEFAULT );
  // dst = equalizeIntensity(dst);
  // dst = equalizeIntensity(dst);
  // dst = equalizeIntensity(dst);
  //Scharr( dst, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
  Sobel( src, grad_x, ddepth, 1, 0, 3, BORDER_DEFAULT );
  //Scharr( dst, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
  Sobel( src, grad_y, ddepth, 0, 1, 3, BORDER_DEFAULT );
  convertScaleAbs( grad_x, abs_grad_x );
  convertScaleAbs( grad_y, abs_grad_y );
  addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
  //Mat channel[3];
  //split(grad, channel);
  cvtColor( grad, grad, COLOR_BGR2GRAY );
  //grad = (channel[0] + channel[1] + channel[2])/3;
  return true;
}

int diff(int a,int b){
  if(a-b >= 0)  return a -b;
  return -1 * (a-b);
}

bool seam_dp(Mat& grad, Mat& seamed_image){
  if(grad.empty())  return -1;

  seamed_image = grad.clone();
  int rows = grad.rows;
  int cols = grad.cols;

  for(int i=0;i<cols;i++)  seamed_image.at<uchar>(0,i) = grad.at<uchar>(0,i);

  uchar left,right,middle;
  for(int i=1;i<rows;i++){
    for(int j=0;j<cols;j++){
      if(j > 0)  left = seamed_image.at<uchar>(i-1,j-1)+diff(grad.at<uchar>(i,j-1),grad.at<uchar>(i,j+1))+diff(grad.at<uchar>(i,j-1),grad.at<uchar>(i-1,j));
      else left = 255;
      if(j < cols-1) right = seamed_image.at<uchar>(i-1,j+1)+diff(grad.at<uchar>(i,j-1),grad.at<uchar>(i,j+1))+diff(grad.at<uchar>(i,j+1),grad.at<uchar>(i-1,j));
      else right = 255;
      middle = seamed_image.at<uchar>(i-1,j)+diff(grad.at<uchar>(i,j-1),grad.at<uchar>(i,j+1));
      seamed_image.at<uchar>(i,j) = seamed_image.at<uchar>(i,j) + min(middle,min(left,right));
    }
  }
  return true;
}

bool remove_col(Mat& output_image,Mat& seamed_image,Mat& colored_image){
  int rows = output_image.rows;
  int cols = output_image.cols;

  Mat reduce_image = Mat(rows, cols-1, CV_8UC3);

  uchar min_point = seamed_image.at<uchar>(rows-1,0);
  int k=0;
  for(int j=1;j<cols;j++){
    if(min_point > seamed_image.at<uchar>(rows-1,j)){
      min_point = seamed_image.at<uchar>(rows-1,j);
      k = j;
    }
  }
  //cout<<k<<endl;
  colored_image.at<Vec3b>(rows-1,k) = Vec3b(255,255,255);
  for(int j=0;j<k;j++) reduce_image.at<Vec3b>(rows-1,j)=output_image.at<Vec3b>(rows-1,j);
  for(int j=k+1;j<cols;j++) reduce_image.at<Vec3b>(rows-1,j-1)=output_image.at<Vec3b>(rows-1,j);

  uchar left,right,middle;
  for(int i=rows-2;i>=0;i--){
    if(k > 0)  left = seamed_image.at<uchar>(i,k-1);
    else left = 255;
    if(k < cols-1) right = seamed_image.at<uchar>(i,k+1);
    else right = 255;
    middle = seamed_image.at<uchar>(i,k);
    uchar new_min = min(middle,min(left,right));
    //cout<<int(left)<<" "<<int(right)<<" "<<int(middle)<<" "<<int(new_min)<<endl;
    if(left == new_min) k = k-1;
    else if(right == new_min) k = k+1;

    colored_image.at<Vec3b>(i,k) = Vec3b(255,255,255);
    // colored_image.at<Vec3b>(i,k)[1] = 255;
    // colored_image.at<Vec3b>(i,k)[2] = 255;
    for(int j=0;j<k;j++) reduce_image.at<Vec3b>(i,j)=output_image.at<Vec3b>(i,j);
    for(int j=k+1;j<cols;j++) reduce_image.at<Vec3b>(i,j-1)=output_image.at<Vec3b>(i,j);
  }

  output_image = reduce_image.clone();
  return true;
}


bool reduce_image(Mat& in_image,Mat& output_image,int new_width,int new_height,Mat& colored_image){
  if(new_width>in_image.cols){
      cout<<"Invalid request!!! new_width has to be smaller than the current size!"<<endl;
      return false;
  }
  if(new_height>in_image.rows){
      cout<<"Invalid request!!! ne_height has to be smaller than the current size!"<<endl;
      return false;
  }

  if(new_width<=0){
      cout<<"Invalid request!!! new_width has to be positive!"<<endl;
      return false;
  }

  if(new_height<=0){
      cout<<"Invalid request!!! new_height has to be positive!"<<endl;
      return false;
  }

  output_image = in_image.clone();
  colored_image = in_image.clone();
  int counter = 0,count = 20;
  int even = 1;
  while(output_image.cols!=new_width || output_image.rows!=new_height){
    if(output_image.cols > new_width){
      Mat seamed_image,grad;
      if(even%2==0){
        rotate(output_image, output_image, 1);
        rotate(colored_image, colored_image, 1);
        // imshow("rotated",output_image);
        // waitKey(0);
      }
      energy_map(output_image,grad);
      if(counter==count){
        //imshow("grad",grad);
        //waitKey(0);
        if(count==20) count = 21;
        else  count = 20;
      }

      seam_dp(grad,seamed_image);
      // if(counter==20){
      //   imshow("seamed_image",seamed_image);
      //   waitKey(0);
      // }

      remove_col(output_image,seamed_image,colored_image);
      // if(counter==20){
      //   imshow("output_image",output_image);
      //   waitKey(0);
      // }

      if(even%2==0){
        rotate(output_image, output_image, 1);
        rotate(colored_image, colored_image, 1);
        // imshow("rotated",output_image);
        // waitKey(0);
      }

      counter++;

    }
    if(output_image.rows > new_height){
      Mat seamed_image,grad,transposed;
      transpose(output_image,transposed);
      transpose(colored_image,colored_image);
      if(even%2==0){
        rotate(transposed, transposed, 1);
        rotate(colored_image, colored_image, 1);
        // imshow("rotated",output_image);
        // waitKey(0);
      }

      energy_map(transposed,grad);
      seam_dp(grad,seamed_image);
      remove_col(transposed,seamed_image,colored_image);
      if(even%2==0){
        rotate(transposed, transposed, 1);
        rotate(colored_image, colored_image, 1);
        // imshow("rotated",output_image);
        // waitKey(0);
      }
      transpose(transposed,output_image);
      transpose(colored_image,colored_image);
    }
            even++;
            if(counter > 21) counter = 0;
  }
  return true;
}


int main( int, char** argv )
{
  Mat src,src_gray;
  Mat grad;
  const char* window_name = "Sobel Demo - Simple Edge Detector";

  src = imread( argv[1], IMREAD_COLOR ); // Load an image
  if( src.empty() )
    { return -1; }


  //GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );
  //cvtColor( src, src_gray, COLOR_BGR2GRAY );


  //if(!energy_map(src_gray,grad)) return -1;
  //imshow( window_name, grad );

  //Mat seamed_image;
  //seam_dp(grad,seamed_image);
  //imshow( "image gradient scored", seamed_image );
  int new_width = atoi(argv[2]);
  int new_height = atoi(argv[3]);
  Mat output_image,colored_image;
  cout<<src.cols<<" "<<src.rows<<endl;
  //imshow( "gray image", src_gray );
  imshow( "input image", src );
  //rotate(src, src, 1);
  //imshow("rotated",src);
  // transpose(src,src);
  // imshow("transposed",src);
  // rotate(src, src, 1);
  // imshow("rotated",src);
  if(!reduce_image(src,output_image,new_width, new_height,colored_image)){
    cout<<"unsuccessful"<<endl;
    return -1;
  }

  cout<<output_image.cols<<endl;
  imshow( "reduced image", output_image );
  imshow( "seamed image", colored_image);
  waitKey(0);
  return 0;
}
