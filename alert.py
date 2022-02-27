from abc import ABC, abstractmethod


class IAlert(ABC):
    """
    This is an interface that needs to be implemented by classes which will be used to send 
    notifications upon detecting objects on the image. This abstract class serves to build an
    Adapter Pattern.
    """
    def __init__(self, from_address, smtp_server_address, smtp_server_port, password):
        """
        Args:
            from_address(str): Adress that alert notification will be sent from.
            smtp_server_address(str): Address of the remote host whom will be connected to send mail.
            smtp_server_port(int): Port of the remote host whom will be connected to send mail.
            password(str): password that will be used to login to from_address.
        """
        super().__init__()
        self.from_address = from_address
        self.smtp_server_address = smtp_server_address
        self.smtp_server_port = smtp_server_port
        self.password = password


    @abstractmethod
    def send(self, to_address, subject, text_content, filename):
        pass


#
import smtplib as smtp
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText  
from email import encoders
from email.header import Header
from email.mime.base import MIMEBase
import cv2 as cv

class MailSender(IAlert):
    """
    This class has functionality to send mails.
    """
    def send(self, to_address, subject, text_content, image, filename):
        # Create MIMEMultipart object. This will be the main message to attach other things to.
        msg = self.instantiate_MIMEMultipart(subject, to_address)
        # Create a MIMEText that will make up the text content of the message in html format 
        msg_content = self.instantiate_MIMEText(text_content)
        # Add the created text content of the mail to the main msg object
        msg.attach(msg_content)

        # Create image content and add the content to the main message
        image_content = self.create_image_attachment(filename, image)
        msg.attach(image_content)

        # Connect to the server and authenticate
        server = self.connect_and_authenticate_to_server()
        # Send the message 
        server.sendmail(self.from_address, [to_address], msg.as_string())
        # Terminate the connection wih the server
        server.quit()

    
    def instantiate_MIMEMultipart(self, subject, to_address):
        msg = MIMEMultipart()
        # Set fields of msg
        msg['From'] = self.from_address
        msg['To'] = to_address
        msg['Subject'] = Header(subject, 'utf-8').encode()
        return msg

    def instantiate_MIMEText(self, text_content):
        msg_content = MIMEText(text_content, 'html', 'utf-8')
        return msg_content

    def create_image_attachment(self, filename, image):
        """
        Creates a MIMEBase object. Takes in RGB image in np.array format. sets the headers filename parameters to inputted filename. Convert image from np.array
        to bytes and attaches it the MIMEBase object. Returns this object.
        Args:
            filename(str): Header's filename parameter will be set to this value.
            image(np.array): RGB image in the format of np.array.
        """
        mime = MIMEBase('image', 'jpg', filename = filename)
        mime.add_header('Content-Disposition', 'attachment', filename = filename)
        mime.add_header('X-Attachment-Id', '0')
        mime.add_header('Content-ID', '<0>')
        # Convert image from RGB to BGR
        image = cv.cvtColor(image,cv.COLOR_RGB2BGR) # Convert image to BGR from RGB
        # Convert image from np.array to image format
        success, encoded_image = cv.imencode('.jpg', image)
        image_bytes = encoded_image.tobytes()
        # Attach image in bytes to the mime object
        mime.set_payload(image_bytes)
        encoders.encode_base64((mime))

        return mime
    
    def connect_and_authenticate_to_server(self):
        """
        Establishes a connection with the smtp server at self.smtp_server_address:self.smtp_server_port.
        Authenticates using self.from_address and self.password. Upon successfull completion returns the SMTP object(server).
        """
        server = smtp.SMTP(self.smtp_server_address, self.smtp_server_port) # Connect to server
        server.starttls() # Enable security
        server.set_debuglevel(0)
        server.login(self.from_address, self.password) # Authenticate the account
        return server


class MMSSender(IAlert):
    """
    This class has functionality to send MMS messages.
    """
    def send(self):
        pass
