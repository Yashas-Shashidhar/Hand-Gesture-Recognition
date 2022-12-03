import splitfolders

splitfolders.ratio("/Users/shreenidhir/Documents/Machine learning/project_18nov/rps_data/", # The location of dataset
                   output="/Users/shreenidhir/Documents/Machine learning/project_18nov/rps_train_test_val_split", # The output location
                   seed=42, # The number of seed
                   ratio=(.7, .2, .1), # The ratio of splited dataset
                   group_prefix=None, # If your dataset contains more than one file like ".jpg", ".pdf", etc
                   move=False # If you choose to move, turn this into True
                   )
