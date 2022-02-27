from alert import MailSender
import cv2 as cv
import queue, threading
from statistics import mode
import numpy as np

from utils.utils import get_current_time
from objectDetector import ObjectDetector



class LiveStream:
    def __init__(self, video_capture_const, hub_model_Url, label_map_path, alertObject, to_address, class_ids=None, train_width=640, train_height=640, fontScale=0.5, thickness=2, box_color=(0, 255, 0), is_box_output_normalized=True, threshold=0.5, displayImage=True, return_original_size=False, sliding_window_size=5):
        """
        Args:
            video_capture_const: This argument is passed into cv2.VideoCapture() function. Can be an index for video capture from
                a camera(e.g. 0 for webcam stream) or can be passed a filepath or rtsp link. For more information visit: https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html
            hub_model_Url(str): Url of the model in TF Hub which is to be used as a model to make predictions on the video frames.
            train_width(int): Width value in int that the TF Hub model used to resize the width of the images to train on(this value
                is specific to the model).
            train_height(int): Height value in int that the TF Hub model used to resize the height of the images to train on(this 
                is specific to the model).
            fontScale(int): Determines the font scale of the text used to annotate the image. Font scale factor that 
                is multiplied by the font-specific base size.
            thickness(int): Determines the thickness of the drawn rectangles and the text put to annotate the image.
            box_color(3 dimensional array): Determines (R,G,B) used to choose the color of the bounding boxes and the annotation text put
                on the image.
            is_box_output_normalized(bool): Should be passed in 'True' if the chosen hub_model returns the detected box coordinates in 'detection_boxes' list
                in a normalized format [between (0-1)]. Pass 'False' if the chosen hub_model returns box coordinates which are not normalized and can be directly
                used to draw boxes.
            label_map_path(str): relative path(in str) to the x_label_map.pbtxt file which is specific to the TF Hub model specified by 
                the hub_model_Url. This file should contain the mappings from prediction indices/id to class labels.
            alertObject(IAlert.class): A concrete class that implements IAlert class in alert.py. This object will be used to send 
                notification messages.
            to_address(str): Address that will received the sent notifications.
            class_ids(list of int): This is a list of ids(int) in the file that label_map_path points to. Predictions with id's
                that are not contained in this list will be filtered out. Passing None as class_ids will result in not filtering
                predictions based on their ids.
            threshold(int): Used to filter predictions. Predictions with prediction_score less than threshold will be disgarded.
            displayImage(bool): If 'True', frames of the stream annotated with boxes and labels will be displayed in a new window.
            return_original_size(bool): If 'True', frames of the stream which are annotated and labeled will be in the size of the
                width and height of the training data used to train the TF Hub model(thes values will be specific to the model used). 
            sliding_window_size(int): size of the sliding window used to determine how many objects are present at a given time. Algorithm keeps track of N many consecutive
            frame's predictions and applies modulo operations per "detected_class"'s in predictions to smoothen wrong and abundant predictions.
        """
        # Important Note!: make sure to call VideoCapture() class here, not the cv.VideoCapture(). 
        # This fixes frames lagging from past instead of being the most recent frame.
        self.cap = VideoCapture(video_capture_const)
        # Create an objectDetector instance to be used as a model to make predictions on the video frames.
        self.objectDetector = self.create_ObjectDetector(hub_model_Url, label_map_path)
        self.class_ids = class_ids
        self.train_width = train_width
        self.train_height = train_height
        self.fontScale = fontScale
        self.thickness = thickness
        self.box_color = box_color
        self.is_box_output_normalized = is_box_output_normalized
        self.threshold = threshold
        self.displayImage = displayImage
        self.return_original_size = return_original_size
        self.N = sliding_window_size
        self.alertObject = alertObject
        self.to_address = to_address

    def show_stream(self):
        """
        Shows the stream on a new window. Press 'q' on your keyboard to close the streaming window.
        """
        while True:
            # Capture frame-by-frame
            ret, frame_bgr = self.cap.read()

            #if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
            # Convert frame from BGR to RGB
            frame_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
            # Display the resulting frame
            cv.imshow('Stream', frame_bgr)
            if cv.waitKey(1) == ord('q'):
                print("Closing the stream window.")
                break
    
    def close_stream(self):
        """
        Close the stream and release the resources allocated to it.
        """
        self.cap.release()
        cv.destroyAllWindows()
            
    def create_ObjectDetector(self, hub_model_Url, label_map_path):
        """
        Creates and returns an instance of ObjectDetector class in objectDetector.py. 
        This returned instance will be used to make predictons on the frames of the stream.

        Args:
            hub_model_Url(str): Url of the model in TF Hub which is to be used as a model to make predictions on the video frames.
            label_map_path(str): Relative path(in str) to the x_label_map.pbtxt file which is specific to the TF Hub model specified by 
                the hub_model_Url. This file should contain the mappings from prediction indices/id to class labels.
        Returns:
            None
        """
               
        # Create an ObjectDetector instance
        return ObjectDetector(hub_model_Url, label_map_path)

    
    def start_object_detector_on_stream(self):
        """
        Reads in the stream from self.cap and uses ObjectDetector to make predictons on these frames.
        Implements a sliding window type algorithm to determine how many objects are present at a given time and should a new notification be sent.
        If a new notification should be sent then, calls classes in alert.py to achieve it.
        """
        # Following are variable initializations used inside the while loop to check if a new notification needs to be sent at any given time.
        sliding_window = [None] * self.N # This list will hold the predictions info btw time [t-N+1, t]
        time_index = 0 # This index will be used to remove the oldest prediction from the sliding_window and add the newest one.
        last_sent_notification_mods = {} # key:"detection_class", value(int): number of instances that "detection_class" was predicted. This holds the last sent notification's info.

        while True:
            # Get the next frame from the stream video in rgb converted np.array format.
            frame_rgb = self.get_next_frame()

            # Make predictions on the current frame
            detector_output = self.predict_on_frame(frame_rgb)
            
            # Filter and format the predictions. Predictions with prediction_score lower than threshold would be disgarded in the process
            predictions_dict = self.objectDetector.filter_and_format_detector_output(detector_output, self.threshold, self.class_ids)
            
            # Get the image annotated with the predictions
            img_with_boxes_rgb = self.objectDetector.annotate_img_with_predictions(frame_rgb, predictions_dict, train_width=self.train_width, 
            train_height=self.train_height, fontScale=self.fontScale, thickness=self.thickness, box_color=self.box_color, is_box_output_normalized=self.is_box_output_normalized, 
            displayImage=False, return_original_size=self.return_original_size)

            # Record the new predictions made and compare them agains the older predictions(in a sliding window way) to prevent
            # unnecessary notificaitons.
            # Get the current predictions grouped according to their "detection_class"
            predictions_classified = self.get_predictions_grouped_by_detection_class(predictions_dict)
            # Add current img_with_boxes_rgb to the dictionary in case, this frame needs to be sent as a notification.
            predictions_classified["image"] = img_with_boxes_rgb

            # Sample predictions_dict below for clarity:
            # {
            #     1:{"number_of_instances": 2},
            #     10:{"number_of_instances": 1},
            #     "image": img_with_boxes_rgb
            # } Here key 1 represents detection_class with label "person", and 2 represents "car".

            # Replace the oldest frame predictions(t-N+1) with the newest frame predictions(t)
            sliding_window[time_index] = predictions_classified
            # update the time_index but make sure it loops to the beginning when it exceeds the sliding_windows array size
            time_index = (time_index + 1) % self.N

            # Check if the indices of the list are initialized(i.e. more than N frames have passed)
            if sliding_window.count(None) == 0:  # if the list is fully initialized we can consider sending a notification
                # For each "detection_class" present in the sliding_window find the mode(i.e modulo) of number_of_instances between the time frames stored in it.
                # If not all of the time frames in the sliding_window have a certain "detection_class" then, count them as an instance with 0 "number_of_instances"
                # while finding its mode.       
                mods = self.get_calculated_modes(sliding_window)
                # Sample mods output:
                    # {
                    #     "detection_class_1": "modulo of class_1",
                    #     "detection_class_10": "modulo of class_10",
                    #     ...
                    # }

                # Compare mods with the last_sent_notification_mods to see if there is a need to send a new notification 
                if (mods != last_sent_notification_mods): # If they are not equal and mods is not empty then, a new notification needs to be sent
                    last_sent_notification_mods = mods.copy()
                    if (bool(mods)):
                        # Update last_sent_notification_mods to be equal to mods since we sent it last now.
                        # Find the image that belongs to the mod
                        for prediction_time_instance in sliding_window: # iterate through each time's grouped predictions in sliding_window
                            prediction_time_instance_clone_without_image = prediction_time_instance.copy()
                            del prediction_time_instance_clone_without_image["image"]
                            if prediction_time_instance_clone_without_image == mods: # mods doesn't have "image" key so compare it with prediction_time_instance.pop("image")
                                mods["image"] = prediction_time_instance["image"]
                        if mods.__contains__("image") == False: # if no image was attached to mods since no prediction with the exact key-valu combinations existed
                            mods["image"] = np.random.choice(sliding_window)["image"] # attach a random image from the sliding_window
                        
                        # Set parameters needed for sending the notification.
                        subject = self.get_alert_subject(mods)
                        text_content = self.get_alert_text_content(mods) 
                        image = mods["image"]
                        filename = "Notification.jpg"
                        # send the notification in a non-blocking way
                        self.send_notification_non_blocking(subject, text_content, image, filename)
            
            # Display the image on a new window if displayImage == 'True'
            if self.displayImage:
                img_with_boxes_bgr = cv.cvtColor(img_with_boxes_rgb,cv.COLOR_RGB2BGR) # Convert image to BGR from RGB
                cv.imshow("Annotated Stream", img_with_boxes_bgr)
                if cv.waitKey(1) == ord('q'):
                    print("Closing the stream window.")
                    break


    def predict_on_frame(self, img):
        """
        Returns the predictions made on the given img in a dictionary format.
        
        Args:
            img(np.array): image as a np.array in the shape of (H, W, 3) with RGB values in the range [0-255].
        """
        # Make predictions on the given image using objectDetector
        return self.objectDetector.make_predictions(img)

    def get_next_frame(self):
        """
        Returns:
            frame_rgb(np.array): The next frame from the video stream in np.array format using self.cap which
                is converted RGB format.
        """
        # Capture frame-by-frame
        ret, frame_bgr = self.cap.read()
        #if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            self.close_stream()
        # Convert frame from BGR to RGB
        frame_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
        return frame_rgb

    def get_predictions_grouped_by_detection_class(self, predictions_dict):
        """
        Takes in predictions_dict and returns a dictionary with key:"detection_class" and value:number of predictions that had the given "detection_class"
        in the predictions_dict.
        
        Args:
            predictions_dict(dict): An array of dictionaries. Each dictionary has a detection_class, name, detection_box and detection_score keys.
        Returns:
            predictions_classified(dict): key:"detection_class" and value:number of predictions that had the given "detection_class"
        in the predictions_dict.

        Sample Output:
            Return {
                    1: 1,
                    10: 3
                    }
                Here key:1 is the "detection_class" with label "person" and it appeared in predictions for only once(value:1),
                 and 10 is the "detection_class" for "car" and it appeared in the predicions for 3(value) times. 
        """
        predictions_classified = {} # dictionary with key: "detection_class", value: how many predictions with the same "detection_class" exists in the prediciton
        for prediction in predictions_dict:
            current_pred_class = prediction["detection_class"]
            if current_pred_class not in predictions_classified.keys(): # if the class is not added before then, initialize its value
                predictions_classified[current_pred_class] = 1
            else: # if the class_id already exists then increment its value("instance_count")
                predictions_classified[current_pred_class] += 1
        return predictions_classified

    def get_calculated_modes(self, sliding_window):
        """
        Given a sliding_window returns a dictionary with key:"detection_class", value(int): modulo(frequency) of the "detection_class" in sliding_window.
        Args:
            sliding_window(dict): key:"detection_class" and value:number of predictions that had the given "detection_class"

        Returns:
            mods(dict): Dictionary with key:"detection_class", value(int): modulo(frequency) of the "detection_class" in sliding_window

        Sample Return:
            Sample mods output:
                    {
                        "detection_class_1": "modulo of class_1",
                        "detection_class_10": "modulo of class_10",
                        ...
                    }
        """
        modded_detection_classes = [] # contains the "detection_class"'es that their modes are calculated.
        mods = {} # key: "detection_class", value:corresponding mode(int). Contains "detection_class"-mode pairs calculated from sliding_window
        
        for prediction_time_instance in sliding_window: # iterate through each time's grouped predictions in sliding_window
            for current_detection_class in prediction_time_instance: # iterate through every "detection_class" in the prediction_time_instance
                if current_detection_class != "image": # Make sure to skip key = "image" since its not a prediction !
                    if current_detection_class not in modded_detection_classes: # make current current_detection_class's mode has not been calculated yet
                        occurence_count = [] #
                        for i in range(len(sliding_window)):
                            current_grouped_preds = sliding_window[i]
                            if current_grouped_preds.__contains__(current_detection_class): # If current detection_class exists in the current_grouped_pred
                                occurence_count.append(current_grouped_preds[current_detection_class])
                            else: # If current detection_class doesn't exist in the current_grouped_pred then append 0
                                occurence_count.append(0)
                        # Calculate the mod using the occurence_count and record the information
                        current_mod = mode(occurence_count)
                        mods[current_detection_class] = current_mod
                        modded_detection_classes.append(current_detection_class)
        # Remove keys with mode 0 from mods
        keys_to_remove = [] #holds keys with value = 0
        for key, value in mods.items():
            if value == 0:
                keys_to_remove.append(key)
        for element in keys_to_remove:
            del mods[element]
        return mods


    def get_alert_subject(self, mods):
        """
        Given mods, formats and returns the information in the mods to a format suitable to be used as alert.send() method's subject parameter.

        Sample Input:
            {
                1: 2,
                10: 1
            }

        Sample Return Output:
            "Alert Notification: [person(1), couch(1)]"
        """
        mods_mapped_with_labels = self.objectDetector.get_labels_dict()
        mods_labeled = {}
        for key in mods:
            if key != "image":
                mods_labeled[mods_mapped_with_labels[key]] = mods[key]
        # Format the information in mods_labeled
        subject ="Alert Notification: ["
        for label, count in mods_labeled.items():
            if label != "image":
                subject += str(label) +"(" + str(count) + "), "
        subject = subject.strip()[:-1] # remove the last comma in the end
        subject += "]"
        return subject

    def get_alert_text_content(self, mods):
        """
        Given mods, formats and returns the information in the mods to a format(html) suitable to be used as alert.send() method's subject parameter.

        Sample Input:
            {
                1: 2,
                10: 1
            }

        Sample Return Output:
            "This is an auto-generated mail to inform you that new objects are Detected: [person(1)]
            2021/08/22, 12:17:28"
        """
        mods_mapped_with_labels = self.objectDetector.get_labels_dict()
        mods_labeled = {}
        for key in mods:
            if key != "image":
                mods_labeled[mods_mapped_with_labels[key]] = mods[key]
        # Format the information in mods_labeled
        detected_string ="Detected: ["
        for label, count in mods_labeled.items():
            if label != "image":
                detected_string += str(label) +"(" + str(count) + "), "
        detected_string = detected_string.strip()[:-1] # remove the last comma in the end
        detected_string += "]"

        text_content = "<html><body> <h2>This is an auto-generated mail to inform you that new objects are <b><em>" + detected_string + \
        "</b></em> <em> <br>"+ get_current_time() +"</em> </h2></body></html>"

        return text_content

    def send_notification_non_blocking(self, subject, text_content, image, filename):
        # Call send() function in another thread to not block this loop.
        t = threading.Thread(target=self.alertObject.send(self.to_address, subject, text_content, image, filename))
        t.daemon = True
        t.start()



class VideoCapture:
    """
    This is used to fix the problem caused by cap = cv.VideoCapture()'s cap.read() returning the next frame from the buffer
    instead of the most recent frame in the stream. This causes LiveStream.start_object_detector_on_stream()'s output annotated
    video and the frames used to make predictions lag behind the recent frames. To fix this instead of cap = cv.VideoCapture(),
    we use cap = VideoCapture(), and call this class's read() method by doing cap.read() in the LiveStream class. This class
    utilizes multi-threading to have a separate thread consume the frames from the buffers constantly which leads the cap.read()
    method to return the most recent frame instead of the buffered next frame as cv.read() does.
    """
    def __init__(self, video_capture_const):
        self.cap = cv.VideoCapture(video_capture_const)
        # Check if stream can be successfully opened
        if not self.cap.isOpened():
            print("Cannot open camera!")
            exit()
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # Read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait() # Discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put((ret, frame))

    def read(self):
        return self.q.get()

    def release(self):
        self.cap.release()
        self.t.terminate()
