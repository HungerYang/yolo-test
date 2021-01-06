#include "class_timer.hpp"
#include "class_detector.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <memory>
#include <thread>

using namespace cv;
using namespace std;

int INPUT_SIZE = 640; //416

void test_demo(const Config &config)
{
	std::unique_ptr<Detector> detector(new Detector());
	detector->init(config);
	cv::Mat image0 = cv::imread("../configs/dog.jpg", cv::IMREAD_UNCHANGED);
	cv::Mat image1 = cv::imread("../configs/person.jpg", cv::IMREAD_UNCHANGED);
	std::vector<BatchResult> batch_res;
	Timer timer;
	for (;;)
	{
		//prepare batch data
		std::vector<cv::Mat> batch_img;
		cv::Mat temp0 = image0.clone();
		cv::Mat temp1 = image1.clone();
		batch_img.push_back(temp0);
		batch_img.push_back(temp1);

		//detect
		timer.reset();
		detector->detect(batch_img, batch_res);
		timer.out("detect");

		//disp
		for (int i=0;i<batch_img.size();++i)
		{
			for (const auto &r : batch_res[i])
			{
				std::cout <<"batch "<<i<< " id:" << r.id << " prob:" << r.prob << " rect:" << r.rect << std::endl;
				cv::rectangle(batch_img[i], r.rect, cv::Scalar(255, 0, 0), 2);
				std::stringstream stream;
				stream << std::fixed << std::setprecision(2) << "id:" << r.id << "  score:" << r.prob;
				cv::putText(batch_img[i], stream.str(), cv::Point(r.rect.x, r.rect.y - 5), 0, 0.5, cv::Scalar(0, 0, 255), 2);
			}
			cv::imshow("image"+std::to_string(i), batch_img[i]);
		}
		cv::waitKey(10);
	}
}


// Darknet weights -> TRT engine.
int weights2trt(const Config &config)
{
    string precision;
    if (config.inference_precison == FP32) 
        precision = "kFLOAT";
    else if (config.inference_precison == FP16) 
        precision = "kHALF";
    else if (config.inference_precison == INT8) 
        precision = "kINT8";
    else {
        std::cout << "Error inference precision in config! Please make sure it should be FP32, FP16 or INT8!" << std::endl;
        return 0;
    }

    std::cout << "Transfer Model from " << config.file_model_weights << " with inference precision " << precision << " to TRT engine." << std::endl;
    std::cout << "...transfering..." << std::endl;
	std::unique_ptr<Detector> detector(new Detector());
	detector->init(config);
	// m_EnginePath = networkInfo.data_path + "-" + m_Precision + "-batch" + std::to_string(m_BatchSize) + ".engine";
	auto npos = config.file_model_weights.find(".weights");
	assert(npos != std::string::npos
        && "wts file file not recognised. File needs to be of '.weights' format");
	std::string weight_path = config.file_model_weights.substr(0, npos);
    std::string engine_name = weight_path + "-" + precision + "-batch" + "1" + ".engine";
    std::cout << "Transfer finished. See file: " << engine_name << std::endl;

    return 0;
}


int test_camera(const Config &config, int dev_id=0)
{
    std::unique_ptr<Detector> detector(new Detector());
    detector->init(config);
    
    // VideoCapture cap(0)
    cv::VideoCapture cap(dev_id);
    if (!cap.isOpened())
    {
        std::cout << "Video device open error! Please check your camera!" << std::endl;
        return -1;
    }

    // Load Camera (batch method)
    std::vector<cv::Mat> batch_img;
    std::vector<BatchResult> batch_res;
    cv::Mat frame;
    Timer timer;
    
    while(true)
    {
        batch_img.clear();
        batch_res.clear();

        // If mutli-frames, then batch-wise method.
        cap >> frame;
        batch_img.push_back(frame);
                
        // detect
        timer.reset();
        detector->detect(batch_img, batch_res);
        timer.out("detect");
               
        // disp        
    	for (int i=0;i<batch_img.size();++i)
        {
            for (const auto &r : batch_res[i])
            {
                std::cout <<"batch "<<i<< " id:" << r.id << " prob:" << r.prob << " rect:" << r.rect << std::endl;
                cv::rectangle(batch_img[i], r.rect, cv::Scalar(255, 0, 0), 2);
                std::stringstream stream;
                stream << std::fixed << std::setprecision(2) << "id:" << r.id << "  score:" << r.prob;
                cv::putText(batch_img[i], stream.str(), cv::Point(r.rect.x, r.rect.y - 5), 0, 0.5, cv::Scalar(0, 0, 255), 2);
            }
            cv::imshow("image"+std::to_string(i), batch_img[i]);
        }
        cv::waitKey(10);
    }     
    return 0;
}

int test_video(const Config &config, const string vid_name)
{
    std::unique_ptr<Detector> detector(new Detector());
    detector->init(config);
    
    cv::VideoCapture cap(vid_name);
    std::cout<< "Reading video file: " << vid_name << std::endl;
    cap.open(vid_name);
    
    if (!cap.isOpened())
    {
        std::cout << "Video file open error! Please check your video path!" << std::endl;
        return -1;
    }

    // Load Video/Camera
    std::vector<cv::Mat> batch_img;
    std::vector<BatchResult> batch_res;
    cv::Mat frame;
    Timer timer;
    
    while(true)
    {
        batch_img.clear();
        batch_res.clear();

        cap >> frame;
        if (frame.empty()) 
            break;

        int cols = frame.cols, rows=frame.rows;
        int dcols, drows;
        if (cols > rows)
        {
            dcols = INPUT_SIZE;
            drows = dcols * rows / cols;
        } else {
            drows = INPUT_SIZE;
            dcols = drows * cols / rows;
        }

        cv::resize(frame, frame, cv::Size(dcols, drows));
        // cv::resize(frame, frame, cv::Size(INPUT_SIZE, INPUT_SIZE));
        batch_img.push_back(frame);
                
        // detect
        timer.reset();
        detector->detect(batch_img, batch_res);
        timer.out("detect");
               
        // disp        
    	for (int i=0;i<batch_img.size();++i)
        {
            for (const auto &r : batch_res[i])
            {
                std::cout <<"batch "<<i<< " id:" << r.id << " prob:" << r.prob << " rect:" << r.rect << std::endl;
                cv::rectangle(batch_img[i], r.rect, cv::Scalar(255, 0, 0), 2);
                std::stringstream stream;
                stream << std::fixed << std::setprecision(2) << "id:" << r.id << "  score:" << r.prob;
                cv::putText(batch_img[i], stream.str(), cv::Point(r.rect.x, r.rect.y - 5), 0, 0.5, cv::Scalar(0, 0, 255), 2);
            }
            cv::imshow("image"+std::to_string(i), batch_img[i]);
        }
        cv::waitKey(10);
    }     
    return 0;
}


const char *keys = 
{
    "{help |  | print this message. }"
    "{h | | print this message}"
    "{model| YOLOV3 | model types. only support YOLOV3, YOLOV3_TINY, YOLOV4, YOLOV4_TINY, YOLOV5}"
    "{pr | FP32 | inference precison. can only be FP32, FP16, INT8. }"
    "{th | 0.5 | detection threshold. should 0< thresh < 1}"
    "{cam | | use camera. }"
    "{vid | ../samples/traffic.mp4 | video file name with full path.}"
    "{trt | | transfer weights models to trt engine, explicitly}"
};

int main(int argc, const char* argv[])
{        
    Config config_v3;
    config_v3.net_type = YOLOV3;
    config_v3.file_model_cfg = "../configs/yolov3.cfg";
    config_v3.file_model_weights = "../configs/yolov3.weights";
    config_v3.calibration_image_list_file_txt = "../configs/calibration_images.txt";
    config_v3.inference_precison =FP32; //FP32;
    config_v3.detect_thresh = 0.5;
    
    Config config_v3_tiny;
    config_v3_tiny.net_type = YOLOV3_TINY;
    config_v3_tiny.detect_thresh = 0.7;
    config_v3_tiny.file_model_cfg = "../configs/yolov3-tiny.cfg";
    config_v3_tiny.file_model_weights = "../configs/yolov3-tiny.weights";
    config_v3_tiny.calibration_image_list_file_txt = "../configs/calibration_images.txt";
    config_v3_tiny.inference_precison = FP32; //FP32;
    
    Config config_v4;
    config_v4.net_type = YOLOV4;
    config_v4.file_model_cfg = "../configs/yolov4.cfg";
    config_v4.file_model_weights = "../configs/yolov4.weights";
    config_v4.calibration_image_list_file_txt = "../configs/calibration_images.txt";
    config_v4.inference_precison =FP32; //FP32; //INT8;
    config_v4.detect_thresh = 0.5;
    
    Config config_v4_tiny;
    config_v4_tiny.net_type = YOLOV4_TINY;
    config_v4_tiny.detect_thresh = 0.5;
    config_v4_tiny.file_model_cfg = "../configs/yolov4-tiny.cfg";
    config_v4_tiny.file_model_weights = "../configs/yolov4-tiny.weights";
    config_v4_tiny.calibration_image_list_file_txt = "../configs/calibration_images.txt";
    config_v4_tiny.inference_precison = FP32; //FP32;
    
    Config config_v5;
    config_v5.net_type = YOLOV5;
    config_v5.detect_thresh = 0.5;
    config_v5.file_model_cfg = "../configs/yolov5-3.0/yolov5s.cfg";
    config_v5.file_model_weights = "../configs/yolov5-3.0/yolov5s.weights";
    config_v5.inference_precison = FP32;
    
    CommandLineParser parser(argc, argv, keys);
    if(parser.has("help") || parser.has("h"))
    {
        parser.printMessage();
        return 0;
    }
    if(!parser.check()) 
    {
        parser.printErrors();
        parser.printMessage();
        return 0;
    }

    string md_type = parser.get<string>("model");
    string precision = parser.get<string>("pr");
    float thresh = parser.get<float>("th");
    bool useCamera = parser.get<bool>("cam");
    string vid_name = parser.get<string>("vid");
    
    if (parser.has("cam"))
        std::cout << "use camera: " << useCamera << std::endl;
    else // if (parser.has("vid"))
        std::cout << "vid_name: " << vid_name << std::endl;
    std::cout << "model type: " << md_type << std::endl;

    Config _config = config_v3;
    if (parser.has("model"))
    {
        if (md_type == "YOLOV3")
            _config = config_v3;
        else if (md_type == "YOLOV3_TINY")
            _config = config_v3_tiny;
        else if (md_type == "YOLOV4")
            _config = config_v4;
        else if (md_type == "YOLOV4_TINY")
            _config = config_v4_tiny;
        else if (md_type == "YOLOV5")
            _config = config_v5;
        else {
            std::cout<< "Error model type input!" << std::endl;
            parser.printMessage();
            return 0;
        }
    }

    if (parser.has("pr"))
    {
        if (precision == "FP32")
            _config.inference_precison = FP32; //FP32; //INT8;
        else if (precision == "FP16")
            _config.inference_precison = FP16; 
        else if (precision == "INT8")
            _config.inference_precison = INT8;
        else {
            std::cout << "Error inference precision input!" << std::endl;
            parser.printMessage();
            return 0;
        }
    }

    if (parser.has("th"))
    {
        if (thresh < 0 || thresh > 1)
        {
            std::cout << "Error threshold input!" << std::endl;
            parser.printMessage();
            return 0;
        }
        _config.detect_thresh = thresh;
    }
    
    // Transfer model from weights to trt engine. 
    if (parser.has("trt") && parser.get<bool>("trt"))
    {
        weights2trt(_config);
        return 0;
    }

    if (parser.has("cam") && useCamera)
        test_camera(_config);
    else //if(parser.has("vid") && vid_name!="")
        test_video(_config, vid_name);
        //test_demo(_config);
    return 0;
}

