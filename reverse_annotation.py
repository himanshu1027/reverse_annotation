import paddleocr
import cv2
from difflib import SequenceMatcher
from flask import Flask, jsonify, request
import os
import json
import re

app = Flask(__name__)

cordinates = None

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def check_similarity(sentence1, sentence2):
    # Normalize sentences (lowercase, remove spaces, etc., as needed)
    normalized_sentence1 = sentence1.lower().replace(" ", "")
    normalized_sentence2 = sentence2.lower().replace(" ", "")

    # Calculate similarity ratio
    similarity_ratio = SequenceMatcher(None, normalized_sentence1, normalized_sentence2).ratio()

    return similarity_ratio

@app.route('/reverse', methods=['POST'])
def annotate_with_exact_text():
    img = request.files['image_data']

    input_text = eval(request.form['data'])

    allowed_extensions = {'.jpg', '.jpeg', '.png' , '.webpg'}
    if img.filename.lower().endswith(tuple(allowed_extensions)):
        file_extension = os.path.splitext(img.filename)[-1]
        filename = os.path.splitext(img.filename)[0]

        project_dir = os.getcwd()

        folder_name = "annotation_images"

        folder_path = os.path.join(project_dir, folder_name)

        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        file_name = f"{filename}{file_extension}"

        file_path = os.path.join(folder_path,file_name)

        img.save(file_path)

        img = cv2.imread(file_path)

        file_name = os.path.basename(file_path)

        if img is not None:
            original_height, original_width, original_channels = img.shape

        ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang='en')

        result = ocr(img)

        paddle_ocr_response = result
        paddle_ocr_text = result[1]

        print(paddle_ocr_text,'paddle_ocr_text')

        confidence_threshold = 0.7

        bounding_demo = {key: [] for key in input_text}
        for i, (text, confidence) in enumerate(paddle_ocr_text):
          if confidence >= confidence_threshold:
            for key,value in input_text.items():
              for data in value:
                similarity = similar(data, text)
                if similarity >= 0.7:
                    bounding_box = paddle_ocr_response[0][i]
                    x, y, w, h = bounding_box[0][0], bounding_box[0][1], bounding_box[2][0], bounding_box[2][1]

                    x, y, w, h = int(x), int(y), int(w), int(h)

                    bounding_demo[key].append((x, y, w, h))
 
                elif key == 'address':
                    # if text in data:
                    # print(text,'text in elif')
                    # print(data,'data in elif')

                    similarity_ratio = similar(text, data)
                    # print(similarity_ratio,'similarity_ratio')

                    # Set a threshold for similarity
                    threshold = 0.4  # You can adjust this value based on your needs

                    # Check if the similarity ratio exceeds the threshold
                    if similarity_ratio >= threshold:
                        # print(text,'text in elif')
                        # print(data,'data in elif')

                        bounding_box = paddle_ocr_response[0][i]
                        x, y, w, h = bounding_box[0][0], bounding_box[0][1], bounding_box[2][0], bounding_box[2][1]

                        x, y, w, h = int(x), int(y), int(w), int(h)

                        bounding_demo[key].append((x, y, w, h))

                        # print("First sentence is similar to a part of the second sentence.")

        data = {
            "img":img,
            "output":bounding_demo
        }

        print(data,'data')

        merged_bounding_box = merge_bounding_boxes(data['output'],img)

        print(merged_bounding_box,'merged_bounding_box')

        cv2.imwrite(f'annotated_{filename}.jpg', merged_bounding_box[0])

        cv_annotation = merged_bounding_box[1]
        image_width = original_width
        image_height = original_height

        yolo_annotation = convert_cv_annotation_to_yolo_annotation(cv_annotation, image_width, image_height)

        file_path = os.path.join(folder_path,filename)

        with open(f"{file_path}.txt", "w") as f:
            for annotation in yolo_annotation:
                f.write(annotation + '\n')

        return yolo_annotation
    else:
       return {"output": "Please provide valid image"}

def merge_bounding_boxes(bounding_boxes,img):

  try:
    final_data = {}
    for key,value in bounding_boxes.items():
        if len(value) > 1:
          x1_min = min([values[0] for values in value])
          y1_min = min([values[1] for values in value])
          x2_max = max([values[2] for values in value])
          y2_max = max([values[3] for values in value])

          x, y, w, h = int(x1_min), int(y1_min), int(x2_max), int(y2_max)
          final_data[key] = [str(x),str(y), str(w), str(h)]
        elif len(value) ==  1:
          data = value[0]

          x, y, w, h = int(data[0]), int(data[1]), int(data[2]), int(data[3])
          final_data[key] = [str(x),str(y), str(w), str(h)]

        cv2.rectangle(img, (x, h), (w, y), (0, 255, 0), 1)
  except:
    pass

  return img,final_data

def convert_cv_annotation_to_yolo_annotation(cv_annotation, image_width, image_height):

  annotation_data = []

  for key,value in cv_annotation.items():
    center_x = (int(value[0]) + int(value[2])) / 2
    center_y = (int(value[1]) + int(value[3])) / 2
    width = int(value[2]) - int(value[0])
    height = int(value[3]) - int(value[1])

    # Normalize the center coordinates and dimensions by the image size.
    normalized_center_x = round(center_x / image_width,6)
    normalized_center_y = round(center_y / image_height,6)
    normalized_width = round(width / image_width,6)
    normalized_height = round(height / image_height,6)

    if key == "title":
      # Write the normalized center coordinates and dimensions to a text file in the YOLO annotation format.
      yolo_annotation = f"{'0'} {normalized_center_x} {normalized_center_y} {normalized_width} {normalized_height}"
    if key == "address":
      # Write the normalized center coordinates and dimensions to a text file in the YOLO annotation format.
      yolo_annotation = f"{'1'} {normalized_center_x} {normalized_center_y} {normalized_width} {normalized_height}"
    if key == "date":
      # Write the normalized center coordinates and dimensions to a text file in the YOLO annotation format.
      yolo_annotation = f"{'2'} {normalized_center_x} {normalized_center_y} {normalized_width} {normalized_height}"
    if key == "orderid":
      # Write the normalized center coordinates and dimensions to a text file in the YOLO annotation format.
      yolo_annotation = f"{'3'} {normalized_center_x} {normalized_center_y} {normalized_width} {normalized_height}"
    if key == "item":
      # Write the normalized center coordinates and dimensions to a text file in the YOLO annotation format.
      yolo_annotation = f"{'4'} {normalized_center_x} {normalized_center_y} {normalized_width} {normalized_height}"
    if key == "tax":
      # Write the normalized center coordinates and dimensions to a text file in the YOLO annotation format.
      yolo_annotation = f"{'5'} {normalized_center_x} {normalized_center_y} {normalized_width} {normalized_height}"
    if key == "taxprice":
      # Write the normalized center coordinates and dimensions to a text file in the YOLO annotation format.
      yolo_annotation = f"{'6'} {normalized_center_x} {normalized_center_y} {normalized_width} {normalized_height}"
    if key == "total":
      # Write the normalized center coordinates and dimensions to a text file in the YOLO annotation format.
      yolo_annotation = f"{'7'} {normalized_center_x} {normalized_center_y} {normalized_width} {normalized_height}"
    if key == "totalprice":
      # Write the normalized center coordinates and dimensions to a text file in the YOLO annotation format.
      yolo_annotation = f"{'8'} {normalized_center_x} {normalized_center_y} {normalized_width} {normalized_height}"

    annotation_data.append(yolo_annotation)

  return annotation_data

# # Example sentences
# sentence1 = "GF-1.MOHNIIOPP.GANDHIGRAM"
# sentence2 = "GF-1,MOH NI  II,OPP.GANDHINAGARAM RAILWAY STATION,ASHRAM ROAD, AHMEDABAD-380009"

# Check similarity

if __name__ == '__main__':
    app.run(debug=True,port=5000)


# input_text = {
#     "title" : ["FOOD INN FINE D"],
#     "totalprice" : ["1838"],
#     "address": ["GF-1,MOH NI  II,OPP.GANDHINAGARAM RAILWAY STATION,ASHRAM ROAD, AHMEDABAD-380009"],
#     "date": ["01/09/2018"],
#     "taxprice" : ["43.75"],
#     "item" : ["LASSI" , "600" , "BUTTER MILK" , "40", "CORN VEGETABLE SC" , "330", "NAVRATAN FORMA","230", "DAL FRIED","320", "RICE(PLAIN)", "130", "ROTI(PLAIN)","100"],
#     "orderid" : ["38391"] 
# }
  
# input_text = {
#     "title" : ["Maui Teriyaki"],
#     "totalprice" : ["14.17"],
#     "address": ["27000 Marine Ave #103 Redondo Beach, CA 90278 (310) 973-3500"],
#     "date": [""],
#     "taxprice" : ["1.23"],
#     "item" : ["Salmon Plate" , "11.49" , "Extra Shrimp" , "1.45"],
#     "orderid" : ["168479"] 
# }

# input_text = {
    # "title" : ["BEN'S BAYSIDE"],
    # "totalprice" : ["41.32"],
    # "address": ["211-37 26th Bayside, NY 11360 Phone: ( 718)229-2367"],
    # "date": ["03/05/17"],
    # "taxprice" : ["3.37"],
    # "item" : ["DOUBLE DIP, 1 tky, bowl barley,  1 past, bowl barley" , " 27.98" , "KNISH" , "0.00", "ROUND" , "3.99", "KASHE SIDE" , "3.99", "COFFEE" , "1.99" , "WATER" , "0.00"],
    # "orderid" : ["45"] 
# }

# input_text = {
#     "title" : ["WILLIAM L"],
#     "totalprice" : ["193.35"],
#     "address": ["DINING ROOM T5 MAIN DINING"],
#     "date": ["03/22/13"],
#     "taxprice" : ["15.35"],
#     "item" : ["BEEFEATER-MARTINI" , "11.50" , "ROSEMARY CAIPIRINHA" , "13.00", "CRABCAKES" , "17.00", "BLACK PEPPER BACON" , "12.00", "Millenium" , "52.00" , "NY SIRLOIN" , "46.00" , "BRUSSEL SPROUTS" , "13.00" , "ESPRESSO" , "4.50" , "LAUREN'S HAZELNUT TORTE" , "9.00"],
#     "orderid" : ["0193"] 
# }


# input_text = {
    # "title" : ["MIGUELS MEXICAN"],
    # "totalprice" : ["27.08"],
    # "address": ["3035 M. KENNEDY BLVD TAMPA, FL."],
    # "date": ["Jul05'17 "],
    # "taxprice" : ["1.78"],
    # "item" : ["WATER" , "0.00" , "ENCH VERDE" , "12.50", "ENCH MANUEL" , "12.80"],
    # "orderid" : [""] 
# }


# input_text = {
    # "title" : ["Bombay Grill House"],
    # "totalprice" : ["118.46"],
    # "address": ["764 9TH AVE NEW YORK, NY 10019 2129771010"],
    # "date": ["30-Jun-2018"],
    # "taxprice" : ["9.66"],
    # "item" : ["Onion Bhajia" , "5.95" , "Lamb Vindaloo" , "14.95", "Konkan Fish Curry" , "16.95", "Shrimp Briyani" , "16.95", "Garlic Naan" , "4.00" , "Hess Cabernet" , "50.00"],
    # "orderid" : ["2"] 
# }

# input_text = {
    # "title" : ["El Meson Mexican Restaurant"],
    # "totalprice" : ["47.70"],
    # "address": ["794 South Perry St Unit EF CASTLE ROCK, CO 80104"],
    # "date": [""],
    # "taxprice" : ["3.50"],
    # "item" : ["SOPA DE TORTILA BOWL" , "5.50" , "TACO SALAD" , "9.50", "HOUSE MARGARITA JUMBO" , "12.50", "Dr Pepper" , "2.65", "SIDE CHEESE" , "1.55" , "HOUSE MARGARITA JUMBO" , "12.50"],
    # "orderid" : ["18533"] 
# }