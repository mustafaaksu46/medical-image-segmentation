1. First, download the "Dresden Abdominal Anatomy Dataset". Extract the Dresden Abdominal Anatomy Dataset from the archive.
2. To prepare the relevant organ in the dataset for processing, you should run either the Patient_Based_Split_Dataset.py or merge_files_by_subdir_id.py Python script.
3. Finally, run the Three_models_Random_Split_train_test_visual_default_DeepLabv3+.py or Three_models_Patient_Based_Split_train_test_visual_default_DeepLabv3+.py script after setting the parameters. These scripts perform all steps—model creation, training, testing, and visualization—as a complete workflow.
4. (Optional) If you want to perform the steps in Step 3 separately, you should first run model.py, then train.py, and finally run either test_visual.py or test_visual_selected_images_detailed.py.
