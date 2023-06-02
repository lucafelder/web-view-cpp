

#include "image_processing.h"

//#define TEST_MODE
#ifdef TEST_MODE
	#define TEST_PICTURE
#endif
//#define uebung09
#define uebung10
#define MEAS_TIME

CImageProcessor::CImageProcessor() {
	for(uint32 i=0; i<3; i++) {
		/* index 0 is 3 channels and indicies 1/2 are 1 channel deep */
		m_proc_image[i] = new cv::Mat();
	}
}

CImageProcessor::~CImageProcessor() {
	for(uint32 i=0; i<3; i++) {
		delete m_proc_image[i];
	}
}

cv::Mat* CImageProcessor::GetProcImage(uint32 i) {
	if(2 < i) {
		i = 2;
	}
	return m_proc_image[i];
}

int CImageProcessor::DoProcess(cv::Mat* image) {

	if(!image) return(EINVALID_PARAMETER);
#ifdef MEAS_TIME
	int64 startTic = cv::getTickCount();
#endif

#ifdef uebung10
	m_net = cv::dnn::readNetFromONNX("/home/pi/CNNdir/MnistCNNBin.onnx");

#ifdef TEST_PICTURE
	*image = cv::imread("/home/pi/CNNdir/MnistRaspiImage_c.png", cv::IMREAD_GRAYSCALE);
#endif // TEST_PICTURE

	cv::Mat grayImage;
	cv::Mat colorImage;
	
	if(image->channels() > 1) {
		cv::cvtColor( *image, grayImage, cv::COLOR_RGB2GRAY );
		colorImage = *image;
	} else {
		grayImage = *image; cv::cvtColor(*image, colorImage, cv::COLOR_GRAY2RGB);
	}

	cv::Mat resultImage = colorImage.clone();

	double threshold1 = 50;
	double threshold2 = 200;
	cv::Mat imgCanny;
	cv::Mat binaryImage;
	cv::Canny(grayImage, imgCanny, threshold1, threshold2, 3, true);
	cv::Mat kernel = cv::Mat::ones(5, 5, CV_8UC1);
	cv::morphologyEx(imgCanny, binaryImage, cv::MORPH_DILATE, kernel);
	cv::Mat stats, centroids, labelImage;
	connectedComponentsWithStats(binaryImage, labelImage, stats, centroids);
	for (int i = 1; i < stats.rows; i++) {
		int topLeftx = stats.at<int>(i, 0);
		int topLefty = stats.at<int>(i, 1);
		int width = stats.at<int>(i, 2);
		int height = stats.at<int>(i, 3);
		int area = stats.at<int>(i, 4);

		// do something
		uint8_t minSize = 20;
		uint8_t maxSize = 40;
		if((minSize <= width || minSize <= height) && width < maxSize && height < maxSize) {
			int sizeCrop = (13*std::max(width, height))/10;
			int topLeftxCrop = std::max(0, topLeftx+(width-sizeCrop)/2);
			int topLeftyCrop = std::max(0, topLefty+(height-sizeCrop)/2);

			int widthCrop = std::min(sizeCrop, binaryImage.cols- topLeftxCrop);
			int heightCrop = std::min(sizeCrop, binaryImage.rows- topLeftyCrop);

			cv::Rect rect(topLeftxCrop, topLeftyCrop, widthCrop, heightCrop);
			cv::rectangle(resultImage, rect, cv::Scalar(255, 0, 0));

			cv::Mat mnistImage = grayImage(rect);
		#ifdef TEST_MODE
			cv::imwrite("mnistImage.png", mnistImage);
		#endif

			double threshold = 0;
			threshold = cv::threshold(mnistImage, mnistImage, threshold, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

			cv::Size classRectSize = cv::Size(28, 28);
			cv::Mat blob = cv::dnn::blobFromImage(mnistImage, 1., classRectSize, cv::Scalar());
			m_net.setInput(blob);
			cv::Mat output = m_net.forward();

			uint8_t maxDigit = 0;
			uint8_t maxValue = 0;

			for(int i0 = 0; i0 < output.cols; i0++) {
				//std::cout << i0 << "," << output.at<float>(0,i0) << std::endl;
				if (maxValue <= output.at<float>(0,i0)){
					maxValue = output.at<float>(0,i0);
					maxDigit = i0;
				}
			}

			//std::cout << maxDigit <<std::endl;
			
			std::string strVal = std::to_string(maxDigit);
			putText(resultImage, strVal.c_str(), cv::Point(topLeftxCrop, topLeftyCrop-5), cv::HersheyFonts::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 0), 2);

		#ifdef TEST_MODE
			cv::imwrite("resultImage.png", resultImage);
		#endif
			
		}
	}

	*m_proc_image[0] = resultImage;

	



#endif // uebung10

#ifdef uebung09

	

	const double threshold = 70;
	static int maxIndex = 50;
	const uint8_t maxIndex_2 = maxIndex/2;
	const int32_t smoothSize = 5;

	// ------------------------------------------------------------
	// 1: Konversion Grauwert-/Farbbild
	// ------------------------------------------------------------

	cv::Mat grayImage;
	cv::Mat colorImage;

	if(image->channels() > 1) {
		cv::cvtColor( *image, grayImage, cv::COLOR_RGB2GRAY );
		colorImage = image->clone();
	} else {
		grayImage = *image;
		cv::cvtColor(*image, colorImage, cv::COLOR_GRAY2RGB);
	}

	// ------------------------------------------------------------
	// 2: Tiefpassfilter
	// ------------------------------------------------------------

	cv::Mat graySmooth;
	cv::blur(grayImage, graySmooth, cv::Size(smoothSize,smoothSize));

	// ------------------------------------------------------------
	// 3/4: partiellen Ableitungen
	// ------------------------------------------------------------

	cv::Mat imgDx, imgDy;
	//cv::Sobel(graySmooth, imgDx, CV_16S, 1, 0, 3, 1, 128);
	//cv::Sobel(graySmooth, imgDy, CV_16S, 0, 1, 3, 1, 128);
	cv::Sobel(graySmooth, imgDx, CV_16S, 1, 0);
	cv::Sobel(graySmooth, imgDy, CV_16S, 0, 1);

	// ------------------------------------------------------------
	// 5: Binning
	// ------------------------------------------------------------

	uint8 colorMap[4][3] = {{255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 0}};

	cv::Mat dirImage(grayImage.size(), CV_8U);
	
	int dims[] = {grayImage.rows, grayImage.cols, 5};
	static cv::Mat bkgrModel(3, dims, CV_8S, maxIndex_2 );

	cv::Mat backgnd(grayImage.size(), CV_8U);
	cv::Mat foregnd_raw(grayImage.size(), CV_8U);
	cv::Mat foregnd(grayImage.size(), CV_8U);
	cv::Mat resultImage(grayImage.size(), CV_8U);

	for(int rows = 0; rows < imgDx.rows; rows++) {

		for(int cols = 0; cols < imgDx.cols; cols++) {

			double dx = imgDx.at<int16_t>(rows, cols);
			double dy = imgDy.at<int16_t>(rows, cols);
			double dr2 = dx*dx + dy*dy;
			int index = 0;

			if(dr2 > threshold*threshold) {
				double alpha = atan2(dy, dx);
				alpha = alpha + M_PI + M_PI_4;
								// shift by pi to convert -pi..pi -> 0..2*pi
    									// shift by pi/4 again for simplified binning
				
				if ( alpha >= (2*M_PI) ) {
					index = 1; // second half of this bin
				} else if ( alpha >= (3*M_PI_2) ) {
					index = 4;
				} else if ( alpha >= M_PI ) {
					index = 3;
				} else if ( alpha >= M_PI_2 ) {
					index = 2;
				} else {
					index = 1; // first half of this bin (no values below pi/4)
				}			
				
				// ------------------------------------------------------------
				// 6: Kantenrichtung in Farbe
				// ------------------------------------------------------------

				colorImage.at<cv::Vec3b>(rows, cols) = cv::Vec3b(colorMap[index - 1]);
			}

			dirImage.at<uint8>(rows, cols) = index;		

			uint8_t maxBkgr = 0;
			uint8_t maxBkgrIndex = 0;

			for(int ind = 0; ind < 5; ind++){
				if (ind == index) {
					if (bkgrModel.at<uint8_t>(rows, cols, ind) < maxIndex) {
						bkgrModel.at<uint8_t>(rows, cols, ind)++;
					}
				} else {
					if ( bkgrModel.at<uint8_t>(rows, cols, ind) > 0 ) {
						bkgrModel.at<uint8_t>(rows, cols, ind)--;
					}
				}

				// ------------------------------------------------------------
				// 7: Vordergrundbild
				// ------------------------------------------------------------
				if ( bkgrModel.at<uint8_t>(rows, cols, ind) > maxIndex_2 && bkgrModel.at<uint8_t>(rows, cols, ind) > maxBkgr ) {
					maxBkgr = bkgrModel.at<uint8_t>(rows, cols, ind);
					maxBkgrIndex = ind;
				}
			} 
			backgnd.at<uint8>(rows, cols) = maxBkgrIndex;
			foregnd_raw.at<uint8>(rows, cols) = dirImage.at<uint8>(rows, cols)!=backgnd.at<uint8>(rows, cols);
		}
	}

	// ------------------------------------------------------------
	// 8: Morphologische Operationen / Merkmalsextraktion
	// ------------------------------------------------------------

	int kernelSize = 5;
	cv::Mat kernel = cv::Mat::ones(kernelSize, kernelSize, CV_8UC1);
	cv::morphologyEx(foregnd_raw, foregnd, cv::MORPH_OPEN, kernel);
	kernelSize = 5;
	kernel = cv::Mat::ones(kernelSize, kernelSize, CV_8UC1);
	cv::morphologyEx(foregnd, foregnd, cv::MORPH_CLOSE, kernel);

	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	double area_min = 5;

	cv::findContours(foregnd, contours, hierarchy, cv::RETR_EXTERNAL , cv::CHAIN_APPROX_SIMPLE);

	resultImage = image->clone();
	
	for(unsigned int idx = 0 ; idx < contours.size(); idx++ ) {
		//area
		double area = cv::contourArea(contours[idx]);

		if ( area >= area_min ) {

			//bounding rectangle
			cv::Rect rect = cv::boundingRect(contours[idx]);

			// center of mass
			cv::Moments moment = cv::moments(contours[idx]);
			double cx = moment.m10 / moment.m00;
			double cy = moment.m01 / moment.m00;

			cv::circle(resultImage, cv::Point(cx, cy), 2, cv::Scalar(0, 0, 255), -1);
			
			//to draw counter to index idx in image
			cv::drawContours(resultImage, contours, idx, cv::Scalar(255), 1, 8 );
		}
	}



	// ------------------------------------------------------------
	// AUSGABE
	// ------------------------------------------------------------

	*m_proc_image[0] = colorImage;
	*m_proc_image[1] = foregnd_raw;
	*m_proc_image[2] = resultImage;

	
#endif

#ifdef MEAS_TIME

	// ------------------------------------------------------------
	// 9: Zeitmessung
	// ------------------------------------------------------------

	int64 endTic = cv::getTickCount();
	double deltaTime = (double) (endTic - startTic)/cv::getTickFrequency();
	std::cout << "time:" << (int) (1000*deltaTime) << " ms" << std::endl;

#endif

	return(SUCCESS);

}
