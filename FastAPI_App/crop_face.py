import face_recognition as fc
import cv2
from PIL import Image, ImageOps

# path = r"G:\Face Recognition with Real-Time\Uploads\22082313.PNG"
# path = "abc.jpg"

def resize_image_with_aspect_ratio(input_path ):
    new_width = 512
    # Open the image
    image = Image.open(input_path)
    image = image.convert("RGB")
    image = ImageOps.exif_transpose(image)
    # Calculate the new height while maintaining the aspect ratio
    aspect_ratio = image.width / image.height
    new_height = int(new_width / aspect_ratio)
    # Resize the image
    #print(new_width, new_height)
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    #print(resized_image.width, resized_image.height)
    # Save the resized image
    resized_image.save(input_path)
    image.close()
    resized_image.close()


# resize_image_with_aspect_ratio(path)
# # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = fc.load_image_file(path,)
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# face_locations = fc.face_locations(img_rgb)  # top, right, bottom, left
# print(face_locations)
# top, right, bottom, left = face_locations[0]
# # Draw a box around the face
# #cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
# crop_img = img_rgb[top:bottom, left:right]
# # cv2.imwrite('test_crop111.png', crop_img)

def crop_face(path ):
    resize_image_with_aspect_ratio(path)
    img = fc.load_image_file(path,)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_locations = fc.face_locations(img_rgb, model='cnn')  # top, right, bottom, left
    print(face_locations)
    top, right, bottom, left = face_locations[0]
    crop_img = img_rgb[top:bottom, left:right]
    cv2.imwrite(path, crop_img)


def crop_face(path ):
    try:
        img = fc.load_image_file(path,)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_locations = fc.face_locations(img_rgb)  # top, right, bottom, left
        print(face_locations)
        top, right, bottom, left = face_locations[0]
        crop_img = img_rgb[top:bottom, left:right]
        cv2.imwrite(path, crop_img)
    except IndexError:
        print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
    return []

def face_recg(path1, path2):
    resize_image_with_aspect_ratio(path1)
    resize_image_with_aspect_ratio(path2)
    img1 = fc.load_image_file(path1)
    img2 = fc.load_image_file(path2)

    try:
        face1_encoding = fc.face_encodings(img1)[0]
        face2_encoding = fc.face_encodings(img2)[0]
    except IndexError:
        print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
        quit()

    known_faces = [
        face1_encoding
    ]

    results = fc.compare_faces(known_faces, face2_encoding)#, tolerance=0.9)
    result_d = fc.face_distance(known_faces, face2_encoding)
    print(results, result_d)
    return results, result_d

path1 = r"sr1.jpg"
# path2 = r"tr4.jpg"
path2 = r"tr3.jpg"
crop_face(path1 )
crop_face(path2 )
face_recg(path1, path2)

# tr1 [True] [0.32488453]
# tr2 [True] [0.32974219]
# tr3 [True] [0.18561478]
# tr4 [True] [0.49410122]
# tr4 [True] [0.44762167]

# ## Define scale factor and window size
# scale_factor = 1.1
# sz1 = img_rgb.shape[1] * 2
# sz2 = img_rgb.shape[0] * 2
# cv2.destroyAllWindows()