import os
import cv2
import json

def process_coco_dataset(train_folder, output_dir):
    target_size = (64, 128)
    crop_id = 0
    
    # Ensure output folders exist
    os.makedirs(os.path.join(output_dir, 'empty'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'occupied'), exist_ok=True)

    json_path = os.path.join(train_folder, '_annotations.coco.json')
    
    # Load the COCO JSON annotations
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
        
    # Create dictionaries to quickly look up category names and image filenames
    categories = {cat['id']: cat['name'].lower() for cat in coco_data['categories']}
    print("Categories found in dataset:", categories)
    
    images = {img['id']: img['file_name'] for img in coco_data['images']}
    
    # Loop through every bounding box annotation in the file
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        category_id = ann['category_id']
        
        # COCO format is already [x_top_left, y_top_left, width, height]
        bbox = ann['bbox'] 
        
        if image_id not in images:
            continue
            
        img_filename = images[image_id]
        img_path = os.path.join(train_folder, img_filename)
        
        if not os.path.exists(img_path):
            continue
            
        frame = cv2.imread(img_path)
        if frame is None:
            continue
            
        # Convert floats to integers for pixel slicing
        x, y, w, h = [int(v) for v in bbox]
        
        # Prevent slicing outside the image boundaries
        x, y = max(0, x), max(0, y)
        roi = frame[y:y+h, x:x+w]
        
        # Standardize: Resize and Grayscale
        if roi.shape[0] > 0 and roi.shape[1] > 0:
            roi_resized = cv2.resize(roi, target_size)
            roi_gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
            
            # Check the category name to sort into the right folder
            cat_name = categories.get(category_id, "")
            
            if 'empty' in cat_name:
                folder = 'empty'
            else:
                folder = 'occupied'
                
            save_path = os.path.join(output_dir, folder, f'spot_{crop_id}.jpg')
            cv2.imwrite(save_path, roi_gray)
            crop_id += 1
            
    print(f"Extraction complete: {crop_id} crops saved.")

# Define paths based on your folder structure
train_folder = './raw_dataset/train'
output_dir = './processed_data'

# Execute
process_coco_dataset(train_folder, output_dir)