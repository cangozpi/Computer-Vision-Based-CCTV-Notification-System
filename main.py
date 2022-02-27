from utils.utils import get_config_json
from liveStream import LiveStream
from alert import MailSender


def main():
    config_path = "./config.json" # Path to config.json file
    # Open and parse the config file
    config_json = get_config_json(config_path) # Returns the json file content in a dictionary

    # Set parameter values from the config file
    video_capture_const = config_json["video_capture_const"]
    hub_model_Url = config_json["hub_model_Url"]
    label_map_path = config_json["label_map_path"]
    class_ids = config_json["class_ids"]
    train_width = config_json["train_width"]
    train_height = config_json["train_height"]
    fontScale = config_json["fontScale"]
    thickness = config_json["thickness"]
    box_color = config_json["box_color"]
    is_box_output_normalized = config_json["is_box_output_normalized"]
    threshold = config_json["threshold"]
    displayImage = config_json["displayImage"]
    return_original_size = config_json["return_original_size"]
    sliding_window_size = config_json["sliding_window_size"]

    # Set parameter values from the config file required to instantiate IAlert.class
    #required information
    from_address = config_json["from_address"]
    to_address = config_json["to_address"]
    smtp_server_address = config_json["smtp_server_address"]
    smtp_server_port = config_json["smtp_server_port"]
    password = config_json["password"]
    
    # instantiate a concrete class implementing IAlert.class to be sent to LiveStream constructor
    mailSender = MailSender(from_address, smtp_server_address, smtp_server_port, password)

    # Instantiate a LiveStream instance
    liveStream = LiveStream(video_capture_const, hub_model_Url, label_map_path, mailSender, to_address, class_ids, train_width, train_height, fontScale,
thickness, box_color, is_box_output_normalized, threshold, displayImage, return_original_size, sliding_window_size)
    # Start making predictions on the stream's frames using the object detection model defined by the ObjectDetector class.
    liveStream.start_object_detector_on_stream()
    # Close the stream and its resources
    liveStream.close_stream()

if __name__ == "__main__":
    main()


# Add documentation to your program !
# Find the most effective TF Hub model to be used.
##TODO: figure out how to load model without using hub.load() but the downloaded files instead.
##TODO: do you need to convert the input image to grayscale to boost the performance ?
##TODO: do you have to resize the image H,W to something like 240,240 to boost the model's performance ?
    #--> TF LITE models colleciton on TF Hub ?
    #--> Faster R-CNN with Resnet-101 V1 Object detection model, trained on COCO 2017 dataset with trainning images scaled to 640x640. 



