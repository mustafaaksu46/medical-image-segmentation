First, download the "Dresden Abdominal Anatomy Dataset". Extract the Dresden Abdominal Anatomy Dataset from the archive.
Run either merge_files_by_subdir_id.py or merge_files_by_subdir_id_image_mask.py Python script while it is in the same folder as the dataset.
At the end of Step 2, the original images and masks are collected in a folder named organ_name_new. If you want the masks and images to be stored in separate folders, the merge_files*.py script should be modified accordingly. If there are any issues, the operations should be performed manually.
Finally, run the DeeplabV3plus_pancreas_train_test_visual.py script after setting the parameters. This script performs all steps—model creation, training, testing, and visualization—as a complete workflow.
If you want to perform the steps in Step 4 separately, you should first run model.py, then train.py, and finally run either test_visual.py or test_visual_selected_images.py.
