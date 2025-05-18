import os
import shutil
import argparse

def copy_images_and_labels(source_images_dir, source_labels_dir, dest_images_dir, dest_labels_dir, image_list=None):
    """
    Copy selected images from source_images_dir to dest_images_dir and their 
    corresponding label files from source_labels_dir to dest_labels_dir.
    
    Parameters:
    source_images_dir (str): Directory containing source images
    source_labels_dir (str): Directory containing source label files
    dest_images_dir (str): Directory where selected images will be copied
    dest_labels_dir (str): Directory where corresponding label files will be copied
    image_list (list, optional): List of image filenames to copy. If None, all images will be copied.
    """
    # Create destination directories if they don't exist
    os.makedirs(dest_images_dir, exist_ok=True)
    os.makedirs(dest_labels_dir, exist_ok=True)
    
    # Get list of images to copy
    if image_list is None:
        # If no specific list is provided, use all images in the source directory
        image_list = [f for f in os.listdir(source_images_dir) 
                     if os.path.isfile(os.path.join(source_images_dir, f)) and 
                     any(f.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp'])]
    
     # Counter for statistics
    copied_pairs = 0
    skipped_empty_labels = 0
    missing_labels = 0
    
    print(f"Found {len(image_list)} images to process")
    
    # Process each image
    for image_filename in image_list:
        image_path = os.path.join(source_images_dir, image_filename)
        
        # Skip if the file doesn't exist (in case image_list contains invalid entries)
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_filename} not found in source directory")
            continue
        
        # Get the base name without extension
        base_name = os.path.splitext(image_filename)[0]
        
        # Look for corresponding label file
        label_filename = f"{base_name}.txt"
        label_path = os.path.join(source_labels_dir, label_filename)
        
        if os.path.exists(label_path):
            # Check if label file is not empty
            if os.path.getsize(label_path) > 0:
                # Copy the image to destination
                shutil.copy2(image_path, os.path.join(dest_images_dir, image_filename))
                
                # Copy the label file to destination
                shutil.copy2(label_path, os.path.join(dest_labels_dir, label_filename))
                
                copied_pairs += 1
            else:
                skipped_empty_labels += 1
                print(f"Skipping {image_filename} - label file exists but is empty")
        else:
            print(f"Warning: No label file found for {image_filename}")
            missing_labels += 1
    
    # Print summary
    print(f"\nSummary:")
    print(f"Copied {copied_pairs} image-label pairs")
    print(f"Skipped {skipped_empty_labels} images with empty label files")
    print(f"Missing labels for {missing_labels} images")

def main():
    # parser = argparse.ArgumentParser(description='Copy selected images and their corresponding label files')
    # parser.add_argument('--source-images', required=True, help='Directory containing source images')
    # parser.add_argument('--source-labels', required=True, help='Directory containing source label files')
    # parser.add_argument('--dest-images', required=True, help='Directory where selected images will be copied')
    # parser.add_argument('--dest-labels', required=True, help='Directory where corresponding label files will be copied')
    # parser.add_argument('--image-list', help='Optional file containing list of image filenames to copy (one per line)')
    
    # args = parser.parse_args()
    
    # # Load list of images if file is provided
    # image_list = None
    # if args.image_list and os.path.exists(args.image_list):
    #     with open(args.image_list, 'r') as f:
    #         image_list = [line.strip() for line in f if line.strip()]
    #     print(f"Loaded {len(image_list)} image filenames from {args.image_list}")

    base_dir = "C:\\Users\\User\\Desktop\\head_data\\overhead_data\\train\\"
    source_img = f"{base_dir}images"
    source_lbl = f"{base_dir}labels"
    dest_img = f"{base_dir}copied_images"
    dest_lbl = f"{base_dir}copied_labels"
    
    copy_images_and_labels(
        source_img,# args.source_images,
        source_lbl,# args.source_labels,
        dest_img,# args.dest_images,
        dest_lbl,# args.dest_labels,
        #image_list
    )

if __name__ == "__main__":
    main()