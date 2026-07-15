# Current Feature
We have to do some change in our DAPT (@s3_dapt). Currently we are doing FULL Domain adaptive pre training. Would like to create an option for PEFT-DAPT where we use LORA adapters and use that for continuous training on our corpus. These is what we need to do:

1. Add new variable in .env.common to toggle PEFT-DAPT
2. If PEFT-DAPT, we need to load LORA adapters into the model and use it for training and inference

<!-- Goals -->

<!-- Notes -->
