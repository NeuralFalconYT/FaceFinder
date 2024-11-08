#modified by: neuralfalcon 
import numpy as np
import cv2
import os 
import shutil

id=1
store_folder="./faces"
if os.path.exists(store_folder):
    shutil.rmtree(store_folder)
os.makedirs(store_folder)
def image_resize(image, width=None, height=None):
    assert width is not None or height is not None, 'width or height must be specified.'

    h, w = image.shape[:2]
    hw_ratio = h/w

    if None in [width, height]:
        if width == None:
            width = height * (1.0 / hw_ratio)
        else:
            height = width * hw_ratio
    
    dsize = (int(width), int(height))
    return cv2.resize(image, dsize=dsize)

def non_max_suppression(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the boxes by
    # their bottom-right y-coordinate
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index
        # value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the
        # bounding box and the smallest (x, y) coordinates for the
        # end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap between the bounding box and
        # other bounding boxes
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap
        # greater than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked
    return boxes[pick], pick

def put_text_on_image(image, text, color=(255, 255, 255)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    position = (10, 50)  # top left corner position

    # Add the text to the image
    cv2.putText(image, text, position, font, font_scale, color, thickness)

    # Convert the image back to BGR if it was originally color
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return image

def hex_to_bgr(hex_color):
    # Remove the '#' if it's there
    hex_color = hex_color.lstrip('#')
    
    # Convert the hex to RGB
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    
    # Convert RGB to BGR
    bgr_color = (b, g, r)
    return bgr_color
def expand_bbox(x1, y1, x2, y2, padding):
    # Increase the size of the bounding box by the padding value
    new_x1 = max(0, x1 - padding)  # Ensure x1 doesn't go below 0
    new_y1 = max(0, y1 - padding)  # Ensure y1 doesn't go below 0
    new_x2 = x2 + padding
    new_y2 = y2 + padding
    
    return new_x1, new_y1, new_x2, new_y2



def resize_if_below_threshold(image, min_width=300, min_height=400):
    height, width = image.shape[:2]  # Get current dimensions

    # Check if the image dimensions are below the specified threshold
    if width < min_width or height < min_height:
        # Calculate new dimensions while maintaining aspect ratio
        aspect_ratio = width / height
        if aspect_ratio > 1:  # Landscape
            new_width = min_width
            new_height = int(min_width / aspect_ratio)
        else:  # Portrait or square
            new_height = min_height
            new_width = int(min_height * aspect_ratio)

        # Resize the image
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        return resized_image
    else:
        print("Image is already equal to or larger than the minimum dimensions.")
    return image  # Return the original image if no resizing is done





def draw_boxes_with_scores(image, boxes, scores,bounding_box=True,display_prediction_labels=False,save=False,save_faces_padding=0,circle_blur_face=False,square_blur_face=False):
    image_fg = image.copy()
    mask_shape = (image.shape[0], image.shape[1], 1)
    mask = np.full(mask_shape, 0, dtype=np.uint8)
    # print(boxes)
    global id
    if len(boxes) == 0:
        return image

    num_boxes = boxes.shape[0]
    for i in range(num_boxes):
        box = boxes[i]
        score = scores[i][0]

        # Convert the box coordinates to integers
        box = box.astype(int)
        #crop face and save it
        try:
            if save:
                x1, y1, x2, y2 = box
                padding = save_faces_padding
                x1, y1, x2, y2 = expand_bbox(x1, y1, x2, y2, padding)
                face=image_fg[y1:y2, x1:x2]
                face=resize_if_below_threshold(face, min_width=300, min_height=400)
                image_name = f"./faces/{id}.jpg"
                print(f"Saving face to {image_name}")
                cv2.imwrite(image_name, face)
                id+=1
        except Exception as e:
            pass

            
        if square_blur_face:
            try:
                pixel_size = 0.1
                x1, y1, x2, y2 = box

                # Ensure the bounding box coordinates are valid
                if x1 >= x2 or y1 >= y2:
                    print("Invalid bounding box coordinates:", box)
                    continue  # Skip this iteration if the box is invalid

                face_img = image[y1:y2, x1:x2].copy()
                
                # Calculate the desired size based on the bounding box
                desired_width = x2 - x1
                desired_height = y2 - y1
                
                # Calculate the small size for resizing
                small_width = int(desired_width * pixel_size)
                small_height = int(desired_height * pixel_size)

                # Ensure the small size is at least 1x1
                small_width = max(1, small_width)
                small_height = max(1, small_height)

                # Resize the face image to a smaller size
                face_img = cv2.resize(
                    face_img,
                    dsize=(small_width, small_height),
                    interpolation=cv2.INTER_NEAREST,
                )
                
                # Resize back to the original size of the bounding box
                face_img_resized = cv2.resize(
                    face_img,
                    dsize=(desired_width, desired_height),
                    interpolation=cv2.INTER_NEAREST,
                )
                
                # Ensure the face image fits exactly in the bounding box
                if face_img_resized.shape[0] != (y2 - y1) or face_img_resized.shape[1] != (x2 - x1):
                    # Create a blank image with the bounding box size
                    blank_face_img = np.zeros((desired_height, desired_width, 3), dtype=np.uint8)

                    # Get the actual width and height of the resized face image
                    actual_height, actual_width = face_img_resized.shape[:2]

                    # Calculate the cropping region if resized image is larger
                    if actual_height > desired_height or actual_width > desired_width:
                        face_img_resized = cv2.resize(face_img_resized, (desired_width, desired_height), interpolation=cv2.INTER_LINEAR)

                    # Place the resized image into the blank image
                    blank_face_img[:actual_height, :actual_width] = face_img_resized

                    # Update face_img_resized to the blank face image
                    face_img_resized = blank_face_img

                # Assign the resized face image to the original image
                image[y1:y2, x1:x2] = face_img_resized
            except Exception as e:
                pass
        if circle_blur_face:
            x1, y1, x2, y2 = box
            # print(x1, y1, x2, y2)

            # Ensure coordinates are within image bounds
            y1 = max(y1, 0)
            y2 = min(y2, image_fg.shape[0])
            x1 = max(x1, 0)
            x2 = min(x2, image_fg.shape[1])

            # Check if the resulting coordinates form a valid region
            if x1 < x2 and y1 < y2:
                w = x2 - x1
                h = y2 - y1

                ksize = (image.shape[0] // 2, image.shape[1] // 2)
                image_fg[y1:y2, x1:x2] = cv2.blur(image_fg[y1:y2, x1:x2], ksize)

                # Calculate the center of the ellipse
                center = ((x1 + x2) // 2, (y1 + y2) // 2)

                # Draw the filled ellipse on the mask
                cv2.ellipse(mask, center, (w // 2, h // 2), 0, 0, 360, 255, -1)
        if circle_blur_face:
            # Combine all masks into the final mask after processing all boxes
            inverse_mask = cv2.bitwise_not(mask)
            image_bg = cv2.bitwise_and(image, image, mask=inverse_mask)
            image_fg = cv2.bitwise_and(image_fg, image_fg, mask=mask)
            image = cv2.add(image_bg, image_fg)
        if bounding_box:
            # Draw the box on the image
            hex_color={"green":"#00ff51","yellow":"#ffd500","cyan":"#00ffff","blue":"#0066ff","red":"#ff0000","white":"#ffffff","light_green":"#00ffae"}
            color=hex_to_bgr(hex_color["green"])
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 1)
            x1, y1, x2, y2 = box
            t=2
            face_width = x2 - x1
            face_height = y2 - y1
            l = min(face_width, face_height) // 5
            # Draw top-left corner
            cv2.line(image, (x1, y1), (x1 + l, y1), color, thickness=t)
            cv2.line(image, (x1, y1), (x1, y1 + l), color, thickness=t)
            # Draw top-right corner
            cv2.line(image, (x2, y1), (x2 - l, y1), color, thickness=t)
            cv2.line(image, (x2, y1), (x2, y1 + l), color, thickness=t)
            # Draw bottom-left corner
            cv2.line(image, (x1, y2), (x1 + l, y2), color, thickness=t)
            cv2.line(image, (x1, y2), (x1, y2 - l), color, thickness=t)
            # Draw bottom-right corner
            cv2.line(image, (x2, y2), (x2 - l, y2), color, thickness=t)
            cv2.line(image, (x2, y2), (x2, y2 - l), color, thickness=t)
            # Define the margin you want between the bounding box and the text
            if display_prediction_labels:
                margin = 5
                thickness = 1
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                text = '{:.2f}'.format(score)
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                # Adjust the text position relative to the box
                text_x = box[0]
                text_y = box[1] - text_size[1] - margin  # Move text up by the margin amount

                # Draw the text background rectangle
                cv2.rectangle(image, (text_x, text_y), (text_x + text_size[0], text_y + text_size[1]), hex_to_bgr(hex_color["blue"]), -1)

                # Draw the text on top of the background rectangle
                cv2.putText(image, text, (text_x, text_y + text_size[1]), font, font_scale, (255,255,255), thickness)
    return image


def crop_image(image, bbox):
    x_min, y_min, x_max, y_max = bbox
    cropped_image = image[y_min:y_max, x_min:x_max]
    return cropped_image
