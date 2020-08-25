# image-classifier

# Usage
To try the default classifier model:
```shell script
python main.py --image_dir "<sub-dir-labeled-image-dir>" \
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
