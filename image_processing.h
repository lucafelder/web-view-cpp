/*! @file image_processing.h
 * @brief Image Manipulation class
 */

#ifndef IMAGE_PROCESSING_H_
#define IMAGE_PROCESSING_H_

#include "opencv.hpp"

#include "includes.h"
#include "camera.h"



class CImageProcessor {
public:
	CImageProcessor();
	~CImageProcessor();
	
	int DoProcess(cv::Mat* image);

	cv::Mat* GetProcImage(uint32 i);
        
        void setStartButton(bool startButton) {m_startButton = startButton;}
	bool getStartButton() const {return m_startButton;}
        
        void setValueMin(int valueMin) {m_valueMin = valueMin;}
	int getValueMin() const {return m_valueMin;}

        void setValueMax(int valueMax) {m_valueMax = valueMax;}
	int getValueMax() const {return m_valueMax;}
        
        void setSaturationMin(int saturationMin) {m_saturationMin = saturationMin;}
	int getSaturationMin() const {return m_saturationMin;}
private:
	cv::Mat* m_proc_image[3];/* we have three processing images for visualization available */
        
        int m_counter;
        
        bool m_startButton;  
        
        int m_valueMin;
        
        int m_valueMax;
        
        int m_saturationMin;
};


#endif /* IMAGE_PROCESSING_H_ */
