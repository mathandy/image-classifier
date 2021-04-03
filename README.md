# image-classifier

# Usage
To try the default classifier model on unsplit directory of 
subdir-labelled images:
```shell script
python main.py --image_dir "<sub-dir-labeled-image-dir>" \
               --logdir "<output_dir>"
```

To try the default classifier model on directory of subdir-labelled 
images already split into train/test or train/test/val directories:
```shell script
python main.py --split_image_dir "<sub-dir-labeled-split-image-dir>" \
               --logdir "<output_dir>"
```

To try many models
```shell script
for m in $(python available_tf_hub_models.py); do   
 	python main.py --image_dir "<sub-dir-labeled-image-dir>" \
                   --logdir "<output_dir>" \
                   --run_name "$m" \
                   --model "$m"
done
```
