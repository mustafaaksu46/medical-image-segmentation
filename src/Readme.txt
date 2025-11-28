1. First, download the "Dresden Abdominal Anatomy Dataset". Extract the Dresden Abdominal Anatomy Dataset from the archive.
2. To prepare the relevant organ in the dataset for processing, you should run either the merge_files_by_subdir_id.py or merge_files_by_subdir_id_image_mask.py Python script.
3. Finally, run the Three_models_train_test_visual_default_DeepLabv3+.py
 script after setting the parameters. This script performs all steps—model creation, training, testing, and visualization—as a complete workflow.
4. (Optional) If you want to perform the steps in Step 3 separately, you should first run model.py, then train.py, and finally run either test_visual.py or test_visual_selected_images_detailed.py.
