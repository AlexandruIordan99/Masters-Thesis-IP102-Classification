# Paths to datasets
import os
import pathlib
import cv2

path_to_train_set = pathlib.PosixPath(
    "/home/jordan/Insect Pest Classification Dataset/classification/train")
path_to_val_set = pathlib.PosixPath(
    "/home/jordan/Insect Pest Classification Dataset/classification/val")
path_to_test_set = pathlib.PosixPath(
    "/home/jordan/Insect Pest Classification Dataset/classification/test")


path_to_train_setbgr = pathlib.PosixPath(
    "/home/jordan/Insect Pest Classification Dataset/classification/trainbgr")
path_to_val_setbgr = pathlib.PosixPath(
    "/home/jordan/Insect Pest Classification Dataset/classification/valbgr")
path_to_test_setbgr = pathlib.PosixPath(
    "/home/jordan/Insect Pest Classification Dataset/classification/testbgr")



def dataset_to_bgr(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    #Loop through folders in a given set
    for class_name in os.listdir(input_folder):
        class_path = os.path.join(input_folder, class_name)
        #Check if the drectory is there
        if os.path.isdir(class_path):
            #Create the relevant class folder
            output_class_path = os.path.join(output_folder, class_name)
            os.makedirs(output_class_path, exist_ok=True)

            for filename in os.listdir(class_path):
                #make sure it's an image
                if filename.endswith((".png", ".jpg", ".jpeg")):
                    img_path = os.path.join(class_path, filename)
                    img = cv2.imread(img_path)
                    if img is not None:
                        #do the actual conversion
                        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        output_path = os.path.join(output_class_path, filename)
                        cv2.imwrite(output_path, img_bgr)
                        print(f"Image successfully converted {output_path}")
                    else:
                        print(f"Image failed to convert {img_path}")

dataset_to_bgr(path_to_train_set, path_to_train_setbgr)

dataset_to_bgr(path_to_val_set, path_to_val_setbgr)

dataset_to_bgr(path_to_test_set, path_to_test_setbgr)
