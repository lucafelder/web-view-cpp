

#include "image_processing.h"


CImageProcessor::CImageProcessor():m_counter(0), m_startButton(false), m_valueMin(20), m_valueMax(255), m_saturationMin(30) {
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
        
    int vmin, vmax, smin;

	vmin = m_valueMin;
	vmax = m_valueMax;
	smin = m_saturationMin;

	if(m_startButton) { 
		std::cout << "vmin: " << vmin << std::endl;
		std::cout << "vmax: " << vmax << std::endl;
		std::cout << "smin: " << smin << std::endl;
		std::cout << "m_counter: " << m_counter << std::endl;

		m_counter++;
	}        

    cv::subtract(cv::Scalar::all(255), *image,*m_proc_image[0]);
        
      //  cv::imwrite("dx.png", *m_proc_image[0]);
      //  cv::imwrite("dy.png", *m_proc_image[1]);

	return(SUCCESS);
}









