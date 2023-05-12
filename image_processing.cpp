

#include "image_processing.h"


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

	cv::Mat grayImage;
	static cv::Mat mPrevImage;	
        
	if(image->channels() > 1) {
		cv::cvtColor( *image, grayImage, cv::COLOR_RGB2GRAY );
	} else {
		grayImage = *image;
	}

	// ------------------------------------------------------------
	// Differenzbild
	// ------------------------------------------------------------

	cv::Mat diffImage;

	if (mPrevImage.size() != cv::Size()) {
		cv::absdiff(mPrevImage, grayImage, diffImage);
		*m_proc_image[0] = diffImage;

		// ------------------------------------------------------------	
		// Binarisierung
		// ------------------------------------------------------------

		cv::Mat binaryImage;
		double threshold =30;

		cv::threshold(diffImage, binaryImage, threshold, 255, cv::THRESH_BINARY);
		*m_proc_image[1] = binaryImage;

		// ------------------------------------------------------------
		// Morphologie
		// ------------------------------------------------------------

		cv::Mat kernel = cv::Mat::ones(5, 5, CV_8UC1);
		cv::morphologyEx(binaryImage, binaryImage, cv::MORPH_CLOSE, kernel);
		*m_proc_image[1] = binaryImage;

		cv::Mat stats, centroids, labelImage;
		connectedComponentsWithStats(binaryImage, labelImage, stats, centroids);
		

		cv::Mat resultImage = image->clone();

		for (int i = 1; i < stats.rows; i++) {
			int topLeftx = stats.at<int>(i, 0);
			int topLefty = stats.at<int>(i, 1);
			int width = stats.at<int>(i, 2);
			int height = stats.at<int>(i, 3);
			int area = stats.at<int>(i, 4);
			double cx = centroids.at<double>(i, 0);
			double cy = centroids.at<double>(i, 1);

			cv::Rect rect(topLeftx, topLefty, width, height);
			cv::rectangle(resultImage, rect, cv::Scalar(255, 0, 0));
			cv::Point2d cent(cx, cy);
			cv::circle(resultImage, cent, 5, cv::Scalar(128, 0, 0), -1);
		}

		*m_proc_image[2] = resultImage;

	}

	mPrevImage = grayImage.clone();

	

	


	return(SUCCESS);
}









