import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import cv2 as cv

from utils.utils import label_map_to_dictionary_converter


class ObjectDetector:
    def __init__(self, tf_hub_Url, label_map_path):
        # Load the model from TF HUB and it's corresponding id-label mapping
        self.detector = hub.load(tf_hub_Url)
        self.labels_dict = self.parse_label_map(label_map_path)
    
    def make_predictions(self, img):
        """
        Takes in an image of type np.array (H,W,3) with with values in the range of [0-255]. Makes predictions on this image and returns a
        dictionary that has information related to predictions made by the model.

        Args:
            img(np.array): np.array of shape (H,W,3) with values in the range of [0-255].
        Returns:
            Returns a dictionary which is specific to the model that encapsulates the information related to the predictions made.
        
        Sample Returned Dictionary:
            key: value -->
            num_detections: a tf.int tensor with only one value, the number of detections [N].
            detection_boxes: a tf.float32 tensor of shape [N, 4] containing bounding box coordinates in the following order: [ymin, xmin, ymax, xmax].
            detection_classes: a tf.int tensor of shape [N] containing detection class index from the label file.
            detection_scores: a tf.float32 tensor of shape [N] containing detection scores.
        """
        # reshape the image to the expected input shape of the loaded model which is 'the input tensor is a tf.uint8 
        # tensor with shape [1, height, width, 3] with values in [0, 255].' 
        img_reshaped =  tf.expand_dims(img, axis=0)

        #make predictions on the image using the model/detector
        detector_output = self.detector(img_reshaped)
        # if detector returns detector_output as a tuple then, convert it to dictionary
        detector_output_dict = {}
        if type(detector_output) == tuple: # if detector_output is tuple then, convert it to dictionary format
            detector_output_dict["detection_boxes"] = detector_output[0]
            detector_output_dict["detection_scores"] = detector_output[1]
            detector_output_dict["detection_classes"] = detector_output[2]
            detector_output_dict["num_detections"] = detector_output[3]
            return detector_output_dict
        return detector_output # if detector_output is already a dictionary then, just return it
        
    def print_detector_output(self, detector_output):
        """
        Takes in detector_output(dictionary that is returned when the model made predictions on the image) and prints the key, 
        value pairs in that dictionary
        
        Args:
            detector_output(dict): dictionary returned by calling the make_predictions(img) method.
        
        Returns:
            None
        """
        for key, value in detector_output.items():
            print("{}: {}\n{}: {} \n".format(key, value, key + ".shape", value.shape))

    def parse_label_map(self, label_map_path):
        """
        Takes in label_map_path which is the relative path to tf hub model's x_label_map_path.pbtxt file and returns a 
        dictionary that contains the id, label mapping using the helper function in utils/label_map_to_dictionary_converter.py
        """
        # Try to map detection_classes to labels using the helper function in the utils
        return label_map_to_dictionary_converter(label_map_path)

    def print_labels_dict(self):
        """
        Prints the contents of self.labels_dict.
        """
        print(self.labels_dict)
    
    def filter_and_format_detector_output(self, detector_output, threshold, class_ids=None):
        """
        Takes in detector_output made by the loaded TF Hub object detection model, threshold(float) and class_ids. Using these and 
        the dictionary of labels with  key:id(int), values:display_name(str) stored in self.labels_dict, returns an array 
        of dictionaries that holds detection_class, name, detection_box, detection_score parameters of the  predictions 
        with higher or equal prediction_score than the threshold value and predictions with id's that doesn't have a match in
        the class_ids list will be filtered out.

        Args:
            detector_output(dict): dictionary that holds predictions returned by the loaded TF Hub model.
            threshold(float): a float value to filter the detection_scores in the detector_output["detection_scores"]
            class_ids(list of int): This is a list of class_ids(int)(equal to detection_class field in the output) that will be used 
                to filter the predictions of the model.Predictions with id's that is not found in this list will be excluded
                from the returned list. Passing None as class_ids will result in not filtering predictions based on their
                ids, and turn off this functionality.
        Returns:
            An array of dictionaries. Each dictionary has a detection_class, name, detection_box and detection_score keys. Every 
            dictionary in the array has a higher or equal detection_score than the inputted threshold value.

        Sample Output:
            [{
                detection_class:1,
                name:"person",
                detection_box:[0.6626782, 0.48261315, 0.8331808, 0.547659],
                detection_score: 0.98
            },{
                detection_class:1,
                name:"person",
                detection_box:[0.5263715, 0.16359192, 0.67855865, 0.24132195],
                detection_score: 0.96
            }]
        """
        # Extract useful parameters from the detector_output dict
        detection_classes = tf.squeeze(detector_output["detection_classes"]).numpy() #shape = (300,)
        detection_boxes = tf.squeeze(detector_output["detection_boxes"]).numpy() #shape = (300, 4)
        detection_scores = tf.squeeze(detector_output["detection_scores"]).numpy() #shape = (300, )
        
        #Filter the elements with corresponding detection_score to have a >= threshold value
        detection_classes_filtered = detection_classes[detection_scores >= threshold]
        detection_boxes_filtered = detection_boxes[detection_scores >= threshold]
        detection_scores_filtered = detection_scores[detection_scores >= threshold]
        # Find corresponding label names using the class id(index)'s
        detection_name_filtered = [self.labels_dict[index] for index in detection_classes_filtered]

        # Create an array of dictionaries that encapsulates these filtered parameters
        # Filter out the predictions with an id that doesn't have a match in the class_ids
        if class_ids != None: # Filter based on detection_class(id)
            predictions_dict_filtered = [{
                "detection_class": detection_classes_filtered[i],
                "name": detection_name_filtered[i],
                "detection_box": detection_boxes_filtered[i],
                "detection_score": detection_scores_filtered[i]
            } for i in range(len(detection_classes_filtered)) if detection_classes_filtered[i] in class_ids]
        else: # Do not filter based on detection_class(id)
            predictions_dict_filtered = [{
            "detection_class": detection_classes_filtered[i],
            "name": detection_name_filtered[i],
            "detection_box": detection_boxes_filtered[i],
            "detection_score": detection_scores_filtered[i]
        } for i in range(len(detection_classes_filtered))]

        return predictions_dict_filtered

    def annotate_img_with_predictions(self, image, predictions_dict, train_width=640, train_height=640, fontScale=0.5, thickness=2, box_color=(0, 255, 0), is_box_output_normalized=True, displayImage=False, return_original_size=True):
        """
        Takes in image(np.array), predictions_dict and returns the image resized to (train_width x train_height) and annotated with
        the predicted class's name, probability and object bounded by a box. Displays the returned image in a new window if
        displayImage = True.

        Args:
            image(np.array): (W,H,3) shaped numpy array that holds [0,255] ranged RGB values of an image. The annotated version of this
                image will be returned.
            predictions_dict: An array of dictionaries. Each dictionary has a detection_class, name, detection_box and detection_score keys.
                Returned by filter_and_format_detection_output(detector_output, labels_dict, threshold) function. Predictions in this array
                will be used to annotate the image.
            train_width(int): Width value in int that the TF Hub model used to resize the width of the images to train on.
            train_height(int): Height value in int that the TF Hub model used to resize the height of the images to train on.
            fontScale(int): Determines the font scale of the text used to annotate the image. Font scale factor that 
                is multiplied by the font-specific base size.
            thickness(int): Determines the thickness of the drawn rectangles and the text put to annotate the image.
            box_color(3 dimensional array): Determines (R,G,B) used to choose the color of the bounding boxes and the annotation text put
                on the image. 
            is_box_output_normalized(bool): Should be passed in 'True' if the chosen hub_model returns the detected box coordinates in 'detection_boxes' list
                in a normalized format [between (0-1)]. Pass 'False' if the chosen hub_model returns box coordinates which are not normalized and can be directly
                used to draw boxes.
            displayImage(bool): Determines whether to plot the returned(annotated with text and bounding boxes) image or not in a new
                window. If 'True' the final version of the image is displayed.
            return_original_size(bool): If 'True' returns the annotated image in the shape of the inputted image. If 'False' 
                returns the annotated image in the size of (train_width, train_height).
        """
        # Save the original image dimension to use them at the end to resize to the original after annotating it
        original_dim = (image.shape[1], image.shape[0]) # (Width, Height)
        # Resize the image before plotting rectangles because loaded tf hub models train on images scaled to train_width X train_height
        # and returns box-coordinates which are normalized according to these dimensions.
        dim = (train_width, train_height)
        img_with_boxes = cv.resize(image, dim) # Resize the image to model's training dimensions
        for prediction_item in predictions_dict:
            # extract information from prediction_item
            name = prediction_item["name"]
            ymin_normalized = prediction_item["detection_box"][0]
            xmin_normalized = prediction_item["detection_box"][1]
            ymax_normalized = prediction_item["detection_box"][2]
            xmax_normalized = prediction_item["detection_box"][3]
            detection_score = prediction_item["detection_score"]

            #if "detection_box" contains normalized coordinates then, convert normalized bounding-box coordinates to normal coordinates
            if is_box_output_normalized:
                ymin = int(ymin_normalized * train_height)
                xmin = int(xmin_normalized * train_width)
                ymax = int(ymax_normalized * train_height)
                xmax = int(xmax_normalized * train_width)
            else: # if "detection_box" contains not normalized coordinates then, they can be directly used to draw boxes.
                ymin = int(ymin_normalized)
                xmin = int(xmin_normalized)
                ymax = int(ymax_normalized)
                xmax = int(xmax_normalized)
            #Add bounding boxes and corresponding annotations to the image
            annotation_string = name + ": " + str(round(detection_score, 2)) + "%"
            img_with_boxes = cv.rectangle(img_with_boxes, (xmin, ymin), (xmax, ymax), box_color, thickness)# plot the surrounding box
            img_with_boxes = cv.putText(img_with_boxes, annotation_string, (xmin, ymin-10), cv.FONT_HERSHEY_SIMPLEX, fontScale, box_color, thickness//2)# plot the annotation text
        #Resize the annotated image back to its original dimensions
        if return_original_size:
            img_with_boxes = cv.resize(img_with_boxes, original_dim)
        # Plot the annotated image
        if displayImage:
            plt.imshow(img_with_boxes)
            plt.axis(False)
            plt.show()
            
        return img_with_boxes

    
    def get_labels_dict(self):
        return self.labels_dict
