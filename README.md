# CCTV Object Detection and Mail Notification Project
## Goal of the project: 
### This project aims to further the capabilities of a _not-smart_ cctv surveillance system by adding object detection and notification sending features.
## How it works :
### It reads in the frames of the video stream. Processes the most current frame with the help of multi-threading. Frames to be processed are passed to Deep Neural Network model to make a prediction. Predictions are filtered via prediction_score and class_labels. Detections on the frame are annotated. It keeps track of the objects in the last sent notification and using a sliding window approach, it calculates the mode(frequency) of the predicted class_labels and how many instances belong to the predicted classes. The calculated frequency is compared to the last sent notifications information and upon comparison if a change is present a new notificaion with the annotated frame from the video is sent.
--- 
## Sample Output:
### * Sample Annotated Frame:
![Sample Annotated Frame](documentation/SampleAnnotatedFrame.jpg?raw=true "Sample Annotated Frame") 
### * Sample Sent Notificaiton Mail:
![Sample Sent Notification](documentation/SampleNotificationMail.jpg?raw=true "Sample Annotated Frame") 
---
---
## How to run:
* Clone the project and cd into the cloned project directory. 
* Open the _config.json_ file. You need to set the parameters for _from_address, to_address, password, smtp_server_address, smtp_server_port_ to configure your notification via mail services. You can learn how to set these parameters via taking a look at the __"How to Configure the Code"__ Section below.
* Then, you can start the project by issuing the following in your preferred terminal.
* Install the following dependencies : tensorflow(2), matplotlib, opencv2, numpy.(The rest should be installed by default with python)
>python3 main.py
* If you want errors to be logged to logs.txt then, run the following
>python3 main.py 2> logs.txt
---
## How to configure the code:
The code is highly configurable. You can tailor it to your own needs and hardware-capabilities by changing a few lines in the **_config.json_** file.
### Parameters in the _Config.json_:
* **__video_capture_const__:** This is the parameter passed to opencv2's cv.VideoCapture() function. Parameters such as _0_,_1_(integer) can be passed to use cameras attached to your own device. To use this project with home cctv camera systems you can pass the rtsp link(e.g "_rtsp://ip:port/sn/live/1/1_") to this parameter as a string.
* **__hub_model_Url__:** This is the Url of the model you want to use copied from TensorFlow Hub website. Pass the url of your model as a string and this will be used to load your model by using tensorflow_hub.load(). Sample Url could be __"https://tfhub.dev/tensorflow/centernet/resnet101v1_fpn_512x512/1"__. There are some things to look for when choosing a model.
1. It should be compatible with TF version 2.
2. The model should provide the following as outputs to a prediction in a dictionary format: "num_detections", __"detection_boxes", "detection_classes", "detection_scores"__.
![Sample Model Output](documentation/SampleModelOutput.png?raw=true "Sample Model Output") 
3. The collection pasted here can be used to pick an already trained object detection model for your needs. __https://tfhub.dev/tensorflow/collections/object_detection/1__. This page is also good in the way that it shows individual model's performance parameters such as __speed__, __mAP__.
* **__label_map_path__:**  This is the path to the label_map.pbtxt file that has the information needed to map the __"detection_class"__ id to its corresponding text label. The default model is trained on mscoco_label_map.pbtxt that can be found default with this project so the path in this case is passed as "./mscoco_label_map.pbtxt" (string). If you want to come up with your own label_map's then you can modify the syntax in that file to your needs. For example if you wanted to add a new class "seagull" with id 100 it would be added as follows: 
>item {
  name: "/m/01g317"
  id: 100
  display_name: "seagull"
}
* **__class_ids__:** This parameter takes in an array as parameter and uses it to filter the predictions made. Passing _null_ leads to not filtering by _detection_class_ in the label_map. If an array is passed in such as _[1, 3]_, this would mean that notifications would be send only for _detection_class_'s with id _1_ and _3_. If we pass msoco_label_map.pbtxt then this would corresponding to only considering predictions that belong to "person" and "car".
* **__train_width__:** This is the width of the images that the Deep Model you've chosen to use expects. Information about this can be found in the model's TF Hub page. For example, if you use model https://tfhub.dev/tensorflow/centernet/resnet101v1_fpn_512x512/1 then, __train_width__ should be 512. Also, by looking at this __train_height__ should be 512 too since model is documented to expect 512x512 images. This parameter is used to scale the frames of the video stream before feeding it to the model to get predictions.
* **__train_height__:** This is the height of the images that the Deep Model you've chosen to use expects. Information about this can be found in the model's TF Hub page. For example, if you use model https://tfhub.dev/tensorflow/centernet/resnet101v1_fpn_512x512/1 then, __train_height__ should be 512. Also, by looking at this __train_width__ should be 512 too since model is documented to expect 512x512 images. This parameter is used to scale the frames of the video stream before feeding it to the model to get predictions.
* **__fontScale__:** This determines the Scale of the font size used to annotate the frames with the text of the corresponding box's class_label and prediction_score. Increase its value to annotate with bigger font size.
* **__thickness__:** This is used to determine the thickness of the borders of the bounding boxes drawn around the predictions in the frames. Increase its value to get thicker borders.
* **__box_color__:** This is a 3-dimensional array that corresponds to RGB values used to determine the color of the bounding boxes drawn around the predicted objects in the frames.
* **__is_box_output_normalized__:**: This is a boolean value. This should be passed in as ```true``` if the chosen hub_model returns the detected box coordinates in 'detection_boxes' list in a normalized format [between (0-1)]. Pass in ```false``` if the chosen hub_model returns not normalized box coordinates which can be directly used to draw boxes.
* **__threshold__:** This parameter takes values between 0 and 1.0 (float). It is used to filter the predictions made on the frames by the model according to the _prediction_score_. Predictions with _prediction_score_ smaller than the passed in threshold value are ignored.
* **__displayImage__:** This takes in boolean values. _True_ displays the frames of the video stream annotated with detected objects on a new window. Passing _False_ results in not displaying the annotated video stream in a new window. If you are running this project just for getting alert notificaitons upon detected objects than I would suggest passing this _False_ to avoid performance issues.
* **__return_original_size__:** If _False_ is passed then the frames annotated with detected objects are in the size of the (__train_width__ x __train_height__) instead of the original size of the video stream. If _True_ is passed then, the frames displayed on the screen and sent as an attachment with the notification are in the same dimension as the original video stream.
* **__sliding_window_size__:** This determines the size of the sliding window used to implement predicton smoothing to prevent faulty notifications stemming from flickering object detections. For example, passing this parameter as 5(int) would result in calculating the mode(frequency) of _detection_class_'s and their corresponding instances in a given frame for the past 5 frames(including the current one) and comparing its results with the last sent notification's values to decide on if a new notificaiton should be sent.
* **__from_address__:** Mail address passed in as a string. This address will be the sender of the notificaiton.
* **__to_address__:** Mail address passed in as a string. This will be the receiver of the notification mail sent.
* **__smtp_server_address__:** This will be specific to the __from_address__ used. It is the address used to connect to the server using SMTP. For gmail it is "smtp.gmail.com", for others you can look it up.
* **__smtp_server_port__:** This is the port number used to connect to server using SMTP. It should be passed in as an integer. For gmail it is _587_. 
* **__password__**: This is the password used to authenticate __to_address__. This should be the password of the account __to_address__ passed in as string.
---
## Overview of classes of the project:
1. **__main.py__:** This file gets the parsed information from _config.json_ file and uses those parameters to instantiate instances of _MailSender.class_ and _LiveStream.class_. Created _MailSender_ instance is passed along with other parameters to instantiate a _LiveStream_ instance. This _LiveStream_ instance's _liveStream.start_object_detector_on_stream()_ function will be called to start the object deteciton on video frames and sending notifications upon need.
2. **__liveStream.py__:** this file has the classes _LiveStream.class_ and _VideoCapture.class_. _VideoCapture.class_ is used to consume the older frames of the video stream using multi-threading to always make predictions on the latest frame. This class was implemented because, while the Deep Learning model was making predictions on a frame time passed and opencv2's implementation used a buffer which fed us the next consecutive frame instead of the most up to date frame. _LiveStream.class_ orchestrates everything and it has the main loop. It reads in the live stream, calls _ObjectDetector.class_ methods to make predictions on the new frame and filter them. After getting the predictons implements a sliding window approach to smoothen the predictions made by the detector. Uses this calculated frequency information to compare it against the last sent notifications predictions and decides on whether a new notificaiton should be sent. If a new notification needs to be sent then calls a concrete class that implements the _IAlert.class_ interface which has the capability to send notifications. At this time, we are using _MailSender.class_ to send mail notifications.
3. **__objectDetector.py__:** This file has the _ObjectDetector.class_ which is used to make predictions given a frame from the video stream. Also, has some helper functions to filter the results according to _threshold_ and _"class_ids" array passed in the _config.json_ file to filter the predictions by the model. Currently it loads the models using TF Hub but, can be modified to be used with any other custom keras model as long as model outputs are in the same format(dictionary with key: value -->
```python 
  return
            {
            "num_detections": a tf.int tensor with only one value, the number of detections [N].
            "detection_boxes": a tf.float32 tensor of shape [N, 4] containing bounding box coordinates in the following order: [ymin, xmin, ymax, xmax].
            "detection_classes": a tf.int tensor of shape [N] containing detection class index from the label file.
            "detection_scores": a tf.float32 tensor of shape [N] containing detection scores.
            }
```
#### ) (dictionary with certain key-value pairs) and a label_map.pbtx file that fits the new custom model in the same format is prepared and its path is passed specified in the config.json file.
4. **__alert.py__:**  This file has the _IAlert.class_ interface and a concrete class that implements it namely, _MailSender.class_. _MailSender.class_ is used to connect with the server to send a notification mail according to the parameters specified in the _config.json_ file. It's _mailSender.send()_ method is used to send a mail notification with subject, text content and an image. If one wishes to implement other means of sending notifications such as using twilio then, one should implement a class that implements _IAlert.class_ which implements its abstract method _send()_. Then this custom IAlert child class can be passed to _LiveStream.class_ constructor instead of _MailSender_ instance and the rest will be taken care of.
5. **__utils.py__:** This file has utility methods called by other classes.
---

