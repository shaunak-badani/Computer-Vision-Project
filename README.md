# Computer Vision project

> Team FourFront

## How to run

- Create a virtual environment and install dependencies
    ```
    python3 -m venv .venv
    source .venv/bin/activate
    make install
    ```
    
- Run training code:
    ```
    make train
    ```

- Run the project / user interface:
    ```
    make run
    ```


# NOTES
SAM is composed of three parts:
1) Image encoder: responsible for processing the image and creating the image embedding. This is the largest component and training it will demand strong GPU. We are NOT fine-tuning it.
2) Prompt encoder: processes input prompt, in our case the input point.
3) Mask decoder: takes the output of the image encoder and prompt encoder and produces the final segmentation masks.

Justifiying less than competitive results
- Training the image encoder would've improved performance much more.
- We had to use the `small` models checkpoint due to compute limitations. Better checkpoints like `large` and `b+` would be much better.