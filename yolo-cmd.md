Usage: yolo [params] 

	--cam
		use camera.
	-h
		print this message
	--help (value:true)
		print this message.
	--model (value:YOLOV5)
		model types. only support YOLOV3, YOLOV3_TINY, YOLOV4, YOLOV4_TINY, YOLOV5
	--pr (value:FP32)
		inference precison. can only be FP32, FP16, INT8.
	--th (value:0.5)
		detection threshold. should 0< thresh < 1
	--vid (value:../samples/traffic.mp4)
		video file name with full path.
